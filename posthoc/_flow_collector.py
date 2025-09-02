class FlowCollector(object):
    
    def __init__(self, model_node):
        self.node = model_node
        self.collects = [[] for _ in range(len(self.node.tracks))]
        
    def clean(self):
        self.collects = [[] for _ in range(len(self.node.tracks))]
        
    def collect(self):
        for idx, track in enumerate(self.node.tracks):
            self.collects[idx].append(track.flow().detach().numpy())
        