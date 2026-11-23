{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   .. automethod:: __init__

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods if item not in pydantic_members or item not in inherited_members %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes if item not in pydantic_members or item not in inherited_members %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% if methods %}
{% for item in methods if item not in pydantic_members or item not in inherited_members %}
{%- if item != '__init__' %}
.. automethod:: {{ objname }}.{{ item }}
{% endif %}
{%- endfor %}
{% endif %}

{% if attributes %}
{% for item in attributes if item not in pydantic_members or item not in inherited_members %}
.. autoattribute:: {{ objname }}.{{ item }}
{%- endfor %}
{% endif %}
