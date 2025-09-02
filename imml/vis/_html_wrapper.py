from html import escape

class HTMLTreeWrapper:
    _style = """
    <style>
        ul, ol {
            margin-left: 1em;
            padding-left: 1em;
        }
        details summary {
            cursor: pointer;
        }
        li {
            margin: 2px 0;
        }
    </style>
    """
    def __init__(self, data):
        if not isinstance(data, (dict, list, tuple)):
            raise ValueError("Only dict or list is supported.")
        self.data = data

    def __repr__(self):
        return repr(self.data)

    def _repr_html_(self):
        return self.render()
    
    def render(self):
        return self._style + self._to_html(self.data)

    def _to_html(self, obj):
        if isinstance(obj, dict):
            html = "<ul>"
            for key, value in obj.items():
                html += "<li>"
                if isinstance(value, (dict, list, tuple)):
                    html += f"<details><summary><strong>{escape(str(key))}</strong></summary>"
                    html += self._to_html(value)
                    html += "</details>"
                else:
                    html += f"<strong>{escape(str(key))}:</strong> {escape(str(value))}"
                html += "</li>"
            html += "</ul>"
            return html
        elif isinstance(obj, (list, tuple)):
            html = "<ol>"
            for idx, item in enumerate(obj):
                html += "<li>"
                if isinstance(item, (dict, list, tuple)):
                    html += f"<details><summary><strong>Item {idx}</strong></summary>"
                    html += self._to_html(item)
                    html += "</details>"
                else:
                    html += f"{escape(str(item))}"
                html += "</li>"
            html += "</ol>"
            return html
        else:
            return escape(str(obj))
