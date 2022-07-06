from jarvis.db.jsonutils import dumpjson,loadjson
d=loadjson('defect_db.json')
mem=[]
for i in d:
  i['E_vac']=i['EF']
  if 'ff_vac' in i:
   i.pop('ff_vac')
  i.pop('EF')
  mem.append(i)
print(len(mem))
print(mem[0])
dumpjson(data=mem,filename='vacancy_db.json')

