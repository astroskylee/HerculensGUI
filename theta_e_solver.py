"""Exact two-image EPL solver for theta_E and ellipticity magnitude."""
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from herculens.MassModel.mass_model import MassModel


class ThetaEllipticitySolver:
    free_names = ('gamma_1', 'epl_position_angle_1', 'shear_strength_1', 'shear_position_angle_1')
    # Stand-in for -inf when the solved (theta_E, |e|) fails validation.  Finite so a
    # single bad particle cannot turn the whole ELBO into inf/nan.
    INVALID_LOG_DENSITY = -1e8

    def __init__(self):
        self.mass = MassModel(['EPL', 'SHEAR'])

    def lens_kwargs(self, q, free):
        theta, ellipticity = q
        gamma, phi, shear, shear_angle, center_x, center_y = free
        return [
            dict(theta_E=theta, gamma=gamma, e1=ellipticity*jnp.cos(2*phi),
                 e2=ellipticity*jnp.sin(2*phi), center_x=center_x, center_y=center_y),
            dict(gamma1=shear*jnp.cos(2*shear_angle), gamma2=shear*jnp.sin(2*shear_angle),
                 ra_0=center_x, dec_0=center_y),
        ]

    def trace(self, q, free, points):
        x, y = self.mass.ray_shooting(points[:, 0], points[:, 1], self.lens_kwargs(q, free))
        return jnp.stack([x, y], axis=1)

    def residual(self, q, free, points):
        beta = self.trace(q, free, points)
        return beta[1] - beta[0]

    @staticmethod
    def linear_solve(jacobian, value):
        # Also used as the `custom_root` tangent rule, so this is what decides whether the
        # solved (theta_E, |e|) has a finite derivative.  A near-singular jacobian yields a
        # huge or non-finite step; returning a zero step instead means a failed solve
        # contributes no gradient rather than a NaN one.  The sample is rejected anyway by
        # the validity factor in `sample_numpyro`.
        good = jnp.all(jnp.isfinite(jacobian)) & (jnp.abs(jnp.linalg.det(jacobian)) > 1e-8)
        solution = jnp.linalg.solve(jnp.where(good, jacobian, jnp.eye(2)), value)
        return jnp.where(good, solution, jnp.zeros_like(solution))

    @classmethod
    def newton(cls, function, initial):
        def cond(state):
            iteration, q = state
            return (iteration < 60) & (jnp.max(jnp.abs(function(q))) > 1e-11)
        def step(state):
            iteration, q = state
            delta = cls.linear_solve(jax.jacfwd(function)(q), function(q))
            delta *= jnp.minimum(1.0, 0.2/jnp.maximum(jnp.linalg.norm(delta), 1e-15))
            trials = q[None, :] - jnp.power(0.5, jnp.arange(9))[:, None]*delta[None, :]
            # Floor the ellipticity strictly above zero: EPL is not differentiable at
            # exactly circular, so a trial landing on 0.0 gives the residual a NaN
            # derivative and poisons the implicit-function gradient of the whole solve.
            trials = jnp.clip(trials, jnp.array([1e-4, 1e-6]), jnp.array([5.0, 0.95]))
            trials = jnp.concatenate([q[None, :], trials])
            losses = jax.vmap(lambda value: jnp.sum(function(value)**2))(trials)
            return iteration + 1, trials[jnp.argmin(jnp.nan_to_num(losses, nan=jnp.inf))]
        return jax.lax.while_loop(cond, step, (0, initial))[1]

    def solve(self, free, points, initial):
        def tangent(function, value):
            return self.linear_solve(jax.jacfwd(function)(jnp.zeros_like(value)), value)
        return jax.lax.custom_root(lambda q: self.residual(q, free, points), initial, self.newton, tangent)

    def diagnostics(self, q, free, points, prior):
        residual = self.residual(q, free, points)
        jacobian = jax.jacfwd(self.residual, 0)(q, free, points)
        # Feed `slogdet` a well-conditioned matrix.  A singular jacobian gives
        # logabsdet = -inf whose VJP is inf, and a later `jnp.where` does not stop
        # that from reaching the ELBO gradient as 0*inf = nan.  Sanitising the
        # input instead keeps both the value and the gradient finite, which is the
        # same guard `linear_solve` already applies.
        regular = jnp.all(jnp.isfinite(jacobian)) & (jnp.abs(jnp.linalg.det(jacobian)) > 1e-12)
        sign, logabsdet = jnp.linalg.slogdet(jnp.where(regular, jacobian, jnp.eye(2)))
        e = q[1]*jnp.array([jnp.cos(2*free[1]), jnp.sin(2*free[1])])
        valid = jnp.all(jnp.isfinite(q)) & (q[0] > prior['theta_low']) & (q[0] < prior['theta_high'])
        valid &= (q[1] > 1e-10) & jnp.all(e > prior['e_low']) & jnp.all(e < prior['e_high'])
        valid &= (jnp.linalg.norm(residual) < 1e-9) & regular
        return valid, residual, logabsdet, sign, e

    def sample_numpyro(self, points, prior, initial):
        gamma = numpyro.sample('gamma_1', dist.Uniform(prior['gamma_low'], prior['gamma_high']).expand([1]).to_event(1))
        angle_prior = dist.Uniform(0.0, jnp.pi)
        phi = numpyro.sample('epl_position_angle_1', angle_prior.expand([1]).to_event(1))
        shear = numpyro.sample('shear_strength_1', dist.Uniform(prior['shear_strength_low'], prior['shear_strength_high']).expand([1]).to_event(1))
        shear_angle = numpyro.sample('shear_position_angle_1', angle_prior.expand([1]).to_event(1))
        center = numpyro.sample('center_1', dist.TruncatedNormal(
            0.0, prior['center_sigma'], low=prior['center_low'], high=prior['center_high']
        ).expand([2]).to_event(1))
        free = jnp.concatenate([jnp.array([gamma[0], phi[0], shear[0], shear_angle[0]]), center])
        q = self.solve(free, points, initial)
        valid, residual, logabsdet, sign, e = self.diagnostics(q, free, points, prior)
        # Evaluate the log-prior on clipped values so it stays finite even when the
        # Newton solve landed outside the prior support.  When `valid` holds the
        # clipping is a no-op; otherwise the result is discarded by the `where`
        # below, and only its (finite) gradient contribution has to be harmless.
        q_prior = jnp.stack([
            jnp.clip(q[0], prior['theta_low'] + 1e-9, prior['theta_high'] - 1e-9),
            jnp.clip(q[1], 1e-10, 0.95),
        ])
        e_prior = jnp.clip(
            q_prior[1]*jnp.array([jnp.cos(2*free[1]), jnp.sin(2*free[1])]),
            prior['e_low'] + 1e-9,
            prior['e_high'] - 1e-9,
        )
        logprior = dist.Uniform(prior['theta_low'], prior['theta_high']).log_prob(q_prior[0])
        logprior += jnp.sum(dist.TruncatedNormal(0.0, prior['e_sigma'], low=prior['e_low'], high=prior['e_high']).log_prob(e_prior))
        logprior += jnp.log(jnp.maximum(2*q_prior[1], 1e-30)) - angle_prior.log_prob(phi[0])
        # A hard -inf makes the whole SVI particle batch -inf, so the reported ELBO
        # stops being usable for chain selection.  A large finite penalty rejects the
        # sample just as firmly while keeping the loss curve readable.
        numpyro.factor('solver_mass_density', jnp.where(valid, logprior-logabsdet, self.INVALID_LOG_DENSITY))
        numpyro.deterministic('theta_E_1', q[:1])
        numpyro.deterministic('e_1', e[:, None])
        numpyro.deterministic('ellipticity_magnitude_1', q[1])
        numpyro.deterministic('axis_ratio_1', (1-q[1])/(1+q[1]))
        numpyro.deterministic('solver_error', jnp.linalg.norm(residual))
        numpyro.deterministic('solver_valid', valid)
        numpyro.deterministic('solver_jacobian_sign', sign)
        numpyro.deterministic('source_beta_common', jnp.mean(self.trace(q, free, points), axis=0))
        return self.lens_kwargs(q, free)

    def free_from_params(self, params):
        return jnp.concatenate([jnp.array([jnp.ravel(params[name])[0] for name in self.free_names]), jnp.ravel(params['center_1'])])

    def from_params(self, params, initial):
        free = self.free_from_params(params)
        points = jnp.stack([params['ra_ps'], params['dec_ps']], axis=1)
        return self.lens_kwargs(self.solve(free, points, initial), free)
