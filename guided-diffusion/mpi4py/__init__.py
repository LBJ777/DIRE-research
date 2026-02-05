class FakeComm:
    def Get_rank(self): return 0
    def Get_size(self): return 1
    def bcast(self, obj, root=0): return obj
    def barrier(self): pass
    @property
    def rank(self): return 0
    @property
    def size(self): return 1

class FakeMPI:
    COMM_WORLD = FakeComm()

# 导出这个伪装对象，让代码以为它是真的 MPI
MPI = FakeMPI()
