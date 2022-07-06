from jarvis.db.jsonutils import loadjson
import glob
for i in glob.glob("JVASP-*"):
  d=loadjson(i)
  opt_atoms=d['alignnff_surf']
  dft_atoms=d['dft_surf']
  print (d['jid'],dft_atoms,opt_atoms)
