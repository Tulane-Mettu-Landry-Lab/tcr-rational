import plotly.express as px
import pandas as pd
from .module_node import ModuleNode

class ModelComponents(object):

    def __init__(self, model_node:ModuleNode):
        self.model_node = model_node
    
    def _table_components(self):
        df = pd.DataFrame([n.path.split('.')+[f'{n.name}({n.module_name})'] for n in self.model_node.leaves])
        return df
    
    @property
    def figure(self):
        df = self._table_components()
        fig = px.treemap(df, path=df.columns)
        fig.update_layout(margin = dict(t=25, l=5, r=5, b=5))
        return fig
    
    @property
    def table(self):
        return self._table_components()
    
    def __repr__(self):
        return repr(self._table_components())
    
    def _repr_html_(self):
        self.figure.show()
        return ''