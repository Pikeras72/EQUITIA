from cerberus import Validator

class ValidadorInsensible(Validator):
    def _validate_allowed(self, allowed_values, field, value):
        """ {'type': 'container'} """
        if isinstance(value, str):
            valor_normalizado = value.lower()
            valores_permitidos = [str(val).lower() for val in allowed_values]
            if valor_normalizado not in valores_permitidos:
                self._error(field, f"Valor no válido '{value}'")
        else:
            # Si no es un string, usar la función original de cerberus
            if value not in allowed_values:
                self._error(field, f"El valor '{value}' no está permitido.")


''' FUNCIÓN ORIGINAL:

    def _validate_allowed(self, allowed_values, field, value):
        """{'type': 'container'}"""
        if isinstance(value, Iterable) and not isinstance(value, _str_type):
            unallowed = tuple(x for x in value if x not in allowed_values)
            if unallowed:
                self._error(field, errors.UNALLOWED_VALUES, unallowed)
        else:
            if value not in allowed_values:
                self._error(field, errors.UNALLOWED_VALUE, value)
'''