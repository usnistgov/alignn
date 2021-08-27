from jarvis.db.figshare import data
import pandas as pd
df=pd.read_csv('qe_tb_indir_gap_predictions.csv')
df[df['out_data']<0]=0
df['error']=abs(df['original']-df['out_data'])
df2=(df[df['error']>0.5])
d=data('qe_tb')
df_tb=pd.DataFrame(d)
df2['jid']=df2['id']
df3=pd.merge(df2,df_tb,on='jid')
df4=df3[~df3.source_folder.str.contains("vol")]
df4['symb_formula']=df4['formula_x']+'_'+df4['spacegroup_number_y'].astype(str)
df5=df4.drop_duplicates('symb_formula')
dft_3d=data('dft_3d')
df_3d=pd.DataFrame(dft_3d)
df2_3d=df_3d[['jid','spg_number','formula']]
df2_3d['symb_formula']=df2_3d['spg_number']+'_'+df2_3d['formula']
df_common=pd.merge(df5,df2_3d,on='symb_formula')
df6=df5[['jid','atoms']]
df6.to_json('qe_tb_todo.json',orient='records', lines=False)

