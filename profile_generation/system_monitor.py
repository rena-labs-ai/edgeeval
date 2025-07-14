from system_profiling.metrics import MetricsTask

class SystemMonitor:
    def __init__(self, results_dir: str):
        """SystemMonitor has following metrics:
         - CPU util
         - CPU memory usage
         - GPU util
         - GPU memory bandwidth"""
        
        self.results_dir = results_dir
    
    def add_ollama_pgid(self, pgid: int):
        self.ollama_metric_task = MetricsTask(pgid=pgid, results_dir=self.results_dir)
    
    def add_browserd_pgid(self, pgid: int):
        self.browserd_metric_task = MetricsTask(pgid=pgid, results_dir=self.results_dir)
    
    def start(self):
        self.ollama_metric_task.start()
        self.browserd_metric_task.start()
    
    def stop(self):
        self.ollama_metric_task.stop()
        self.browserd_metric_task.stop()

    def dump_csv(self, results_dir: str):
        print(f"TODO: how to dump csv?")
        print(self.ollama_metric_task.get_summary())
        print(self.browserd_metric_task.get_summary())
