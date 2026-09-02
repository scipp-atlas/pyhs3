{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   .. automethod:: __init__

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% if methods %}
{% for item in methods %}
{%- if item != '__init__' %}
.. automethod:: {{ objname }}.{{ item }}
{% endif %}
{%- endfor %}
{% endif %}

{% if attributes %}
{% for item in attributes %}
.. autoattribute:: {{ objname }}.{{ item }}
{%- endfor %}
{% endif %}
