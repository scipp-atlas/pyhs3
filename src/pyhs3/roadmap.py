
class Domain: ... # base class

class ProductDomain(Domain):
    hs3_type: "product_domain"
    ...

class ParameterPoints: ... # base class


Workspace:
    domains:
        - <ProductDomain("default_domain") objeect at 0x1230213>
        - <ProductDomain("nondefault_domain") objeect at 0x1230213>
    parameter_points:
        - <ParameterPoints("default_values") object at 0x123123>

    def pdf(self, ..., ..., domain=0, parameter_points=0): ...


something.pdf()
something.pdf(domain=0)
something.pdf(domain="default_domain")
something.pdf(domain="nondefault_domain")
something.pdf(domain=1)
something.pdf(domain=DomainObject)

    if isinstance(domain, [int, str]):
        domain_obj = self.get_domain(domain)
    elif isinstance(domain, [Domain]):
        domain_obj = domain

class Normal:
    hs3_type = "gaussian_dist"
    def __init__(self, *, mean, sigma): ...
    def pdf(self, *, x): ...
    def logpdf(self, *, x): ...
