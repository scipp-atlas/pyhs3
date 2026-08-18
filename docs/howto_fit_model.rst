.. diataxis: how-to
.. status: implemented

Fit a Model with Optimistix
=============================

You have a jaxified ``model.log_prob`` (see :doc:`howto_evaluate_model`'s
"Evaluate a joint likelihood for a fit") and need the parameter values that
minimize it. pyhs3 doesn't ship a minimizer; this shows the pattern for
wiring one in with `optimistix <https://github.com/patrick-kidger/optimistix>`_,
a JAX-native minimization library. Requires ``pip install pyhs3[jax]
optimistix``.

Build the NLL callable
-------------------------

Continuing from the fit-region ``Analysis`` in :doc:`howto_evaluate_model`,
``fit_model.free_params`` gives the starting values and their names, and
``fit_model.data`` the observed arrays; both are needed on every call. Wrap
the jaxified NLL in a function that takes a single positional array (the
shape optimistix minimizes over) and returns a true scalar:

.. code-block:: python

    import jax.numpy as jnp
    import optimistix as optx

    param_names = sorted(fit_model.free_params)
    y0 = jnp.array([fit_model.free_params[name] for name in param_names])


    def nll(values, args):
        del args  # required by optimistix's signature; unused here
        kwargs = dict(zip(param_names, values, strict=True))
        return nll_graph(**fit_model.data, **kwargs)[0].squeeze()

``nll_graph`` returns a shape-``(1,)`` array (:attr:`~pyhs3.Model.log_prob`'s
own shape convention, see :doc:`model_reference`); ``.squeeze()`` reduces it
to the true scalar optimistix requires, or it raises
``minimisation function must output a single floating-point scalar``.

Minimize
--------

.. code-block:: python

    solver = optx.BFGS(rtol=1e-6, atol=1e-6)
    sol = optx.minimise(nll, solver, y0=y0, max_steps=1000)

    converged = sol.result == optx.RESULTS.successful
    best_fit = dict(zip(param_names, sol.value, strict=True))
    print(f"converged: {converged}")
    print(f"best fit: {best_fit}")
    print(f"-2 log L at best fit: {float(nll(sol.value, None)):.4f}")

For the single-Gaussian example workspace built in :doc:`howto_evaluate_model`,
this converges to ``mu`` at the sample mean of ``observed_x`` and ``sigma`` at
its population standard deviation, as expected for a Gaussian's maximum
likelihood estimate.

Profile a parameter of interest
----------------------------------

A profile scan repeats the minimization once per fixed value of a parameter
of interest, minimizing over everything else. Fix the parameter by baking it
into ``nll`` instead of including it in ``y0``:

.. code-block:: python

    nuisance_names = sorted(fit_model.free_params.keys() - {"mu"})
    nuisance_y0 = jnp.array([fit_model.free_params[name] for name in nuisance_names])


    def profile_nll(nuisance_values, mu_value):
        kwargs = dict(zip(nuisance_names, nuisance_values, strict=True))
        kwargs["mu"] = mu_value
        return nll_graph(**fit_model.data, **kwargs)[0].squeeze()


    scan_points = [-1.0, 0.0, 1.0, 2.0]
    scan_nll = []
    for mu in scan_points:
        sol = optx.minimise(
            profile_nll, solver, y0=nuisance_y0, args=jnp.asarray(mu), max_steps=1000
        )
        scan_nll.append(float(profile_nll(sol.value, jnp.asarray(mu))))

``args`` carries the fixed value through to every evaluation without
retracing the JIT-compiled function for each scan point — the value changes
every scan step, so it stays a JAX-level input rather than a PyTensor
constant baked into the compiled graph. See :doc:`model_reference` for how a
``const=True`` parameter is baked in at construction time instead, which is
the right choice only for a value that never changes across an entire run.
