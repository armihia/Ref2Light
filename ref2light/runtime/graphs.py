import torch


class FrozenCall:
    def __init__(self, function, enabled=True):
        self.function, self.enabled = function, enabled
        self.signature = self.graph = None
        self.replays = 0

    @torch.no_grad()
    def __call__(self, *values):
        if not self.enabled or values[0].device.type != 'cuda':
            return self.function(*values)
        signature = tuple((tuple(v.shape), v.dtype, v.device) for v in values)
        if signature != self.signature:
            self.static = tuple(v.detach().clone() for v in values)
            current = torch.cuda.current_stream(values[0].device)
            stream = torch.cuda.Stream(device=values[0].device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.function(*self.static)
            current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                self.outputs = self.function(*self.static)
            current.wait_stream(stream)
            self.graph, self.signature = graph, signature
        for static, value in zip(self.static, values):
            static.copy_(value)
        self.graph.replay()
        self.replays += 1
        # Returned tensors share graph buffers; clone values that must survive later calls.
        return self.outputs

