.. diataxis: how-to
.. status: implemented

Fit a Model with Optimistix
=============================

You have a jaxified ``model.log_prob`` (see :doc:`/howto/evaluate_model`'s
"Evaluate a joint likelihood for a fit") and need the parameter values that
minimize it. pyhs3 doesn't ship a minimizer; this shows the pattern for
wiring one in with `optimistix <https://github.com/patrick-kidger/optimistix>`_,
a JAX-native minimization library. Requires ``pip install pyhs3[jax]
optimistix``.

Build the NLL callable
-------------------------

Continuing from the fit-region ``Analysis`` in :doc:`/howto/evaluate_model`,
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
own shape convention); ``.squeeze()`` reduces it to the true scalar
optimistix requires, or it raises
``minimisation function must output a single floating-point scalar``.

Minimize
--------

.. code-block:: python

    solver = optx.BFGS(rtol=1e-6, atol=1e-6)
    sol = optx.minimise(nll, solver, y0=y0, max_steps=1000, throw=False)

    if sol.result != optx.RESULTS.successful:
        print(f"did not converge: {sol.result}")
    else:
        best_fit = dict(zip(param_names, sol.value, strict=True))
        print(f"best fit: {best_fit}")
        print(f"-2 log L at best fit: {float(nll(sol.value, None)):.4f}")

``throw=False`` makes ``optx.minimise`` return a ``Solution`` on failure
instead of raising (the default, ``throw=True``, would raise before you
ever see ``sol.result``) — check ``sol.result`` before trusting ``sol.value``,
since it can be meaningless when the solve didn't converge. For the
single-Gaussian example workspace built in :doc:`/howto/evaluate_model`,
this converges to ``mu`` at the sample mean of ``observed_x`` and ``sigma``
at its population standard deviation, as expected for a Gaussian's maximum
likelihood estimate.

Profile a parameter of interest
----------------------------------

A profile scan repeats the minimization once per fixed value of a parameter
of interest, minimizing over everything else. Fix the parameter by baking it
into ``nll`` as a second argument instead of including it in ``y0``:

.. code-block:: python

    nuisance_names = sorted(fit_model.free_params.keys() - {"mu"})
    nuisance_y0 = jnp.array([fit_model.free_params[name] for name in nuisance_names])


    def profile_nll(nuisance_values, mu_value):
        kwargs = dict(zip(nuisance_names, nuisance_values, strict=True))
        kwargs["mu"] = mu_value
        return nll_graph(**fit_model.data, **kwargs)[0].squeeze()

``mu_value`` carries the fixed value into ``profile_nll`` without retracing
the JIT-compiled function for each scan point — it stays a JAX-level input
rather than a PyTensor constant baked into the compiled graph, since it
changes every scan step. A truly constant parameter (one that never changes
across an entire run) is better handled by setting ``const=True`` on its
:class:`~pyhs3.parameter_points.ParameterPoint` instead, which bakes it into
the compiled graph itself.

Batch the whole scan with ``vmap``
--------------------------------------

Minimizing once per scan point in a Python loop works, but ``optx.minimise``
is itself a pure JAX function, so the entire scan — including the
minimization at each point — can be vectorized with ``jax.vmap``:

.. code-block:: python

    import jax

    scan_points = jnp.array([-1.0, 0.0, 1.0, 2.0])


    def fit_at(mu_value):
        sol = optx.minimise(
            profile_nll, solver, y0=nuisance_y0, args=mu_value, max_steps=1000, throw=False
        )
        return profile_nll(sol.value, mu_value)


    scan_nll = jax.vmap(fit_at)(scan_points)

This gives the same result as calling ``fit_at`` in a Python loop over
``scan_points``, in one traced, batched call. ``throw=False`` keeps one
non-convergent scan point from raising for the whole batch; ``fit_at`` above
doesn't check ``sol.result`` per point, but the batched ``sol`` from a call
made outside ``vmap`` would carry ``result`` as an array with one entry per
scan point, checkable the same way as the single-fit example above.
