import glob, os
home = os.path.expanduser("~")
c = sorted({os.path.realpath(p) for p in glob.glob(f"{home}/results/dinomaly_split_carpet*/**/weights/lightning/model.ckpt", recursive=True)})
assert len(c) == 1, c
from anomalib.engine import Engine
from anomalib.models import Dinomaly
m = Dinomaly.load_from_checkpoint(c[0])
m.post_processor = None
p = Engine(default_root_dir=f"{home}/results/peek").predict(model=m, data_path=f"{home}/datasets/MVTec_split/carpet/test_defect")
b = p[0]
print("MAPSHAPE", tuple(b.anomaly_map.shape), "MAPMIN", float(b.anomaly_map.min()), "MAPMAX", float(b.anomaly_map.max()))
print("IMAGESHAPE", tuple(b.image.shape))
print("SCORE_vs_MAXMAP", float(b.pred_score[0]), float(b.anomaly_map[0].max()))
print("NBATCHES", len(p))
