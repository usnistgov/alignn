from jarvis.db.jsonutils import loadjson
import glob
for i in glob.glob("JVASP-*"):
  d=loadjson(i)
  opt_atoms=d['alignnff_evac']
  dft_atoms=d['dft_evac']
  print (d['jid'],dft_atoms,opt_atoms)
