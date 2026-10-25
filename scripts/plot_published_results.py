"""Plot published aggregate matched-assay results; no model fitting is performed."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(input_path,output):
    d=pd.read_csv(input_path);models=d['Source learner'].drop_duplicates().tolist()
    methods=['Simple logit calibration','Empirical-Bayes calibration','Influence-constrained calibration']
    colors=['#B9D2A8','#C1DCEB','#E9BBC4']
    plt.rcParams.update({'font.family':'Arial','font.size':11,'axes.labelsize':12,'axes.labelweight':'bold','pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(14,13))
    y=np.arange(len(models));lookup={(m,c):d[(d['Source learner']==m)&(d['Calibrator']==c)].iloc[0] for m in models for c in methods}
    for ax,metric in zip(axes[0],['Uptake RMSE (mg/g)','Log-uptake RMSE']):
        for j,(method,color) in enumerate(zip(methods,colors)):
            values=[lookup[(m,method)][metric] for m in models]
            ax.scatter(values,y+(j-1)*.18,s=35,color=color,edgecolor='#404040',linewidth=.5,label=method)
        ax.set_xlabel(metric);ax.set_yticks(y,models);ax.invert_yaxis()
    for ax,comparator in zip(axes[1],methods[:2]):
        for metric,color,shift in [('Uptake RMSE (mg/g)',colors[1],-.12),('Log-uptake RMSE',colors[2],.12)]:
            values=[lookup[(m,methods[2])][metric]/lookup[(m,comparator)][metric] for m in models]
            ax.scatter(values,y+shift,s=35,color=color,edgecolor='#404040',linewidth=.5,label=metric)
        ax.axvline(1,color='#454545',ls='--',lw=1);ax.set_yticks(y,models);ax.invert_yaxis();ax.set_xlabel('Constrained error / '+comparator.lower())
    handles,labels=axes[0,0].get_legend_handles_labels();handles2,labels2=axes[1,0].get_legend_handles_labels()
    fig.legend(handles+handles2,labels+labels2,loc='lower center',ncol=3,frameon=False)
    fig.subplots_adjust(left=.16,right=.99,top=.99,bottom=.09,wspace=.48,hspace=.2)
    output.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ['.png','.pdf']:fig.savefig(output.with_suffix(suffix),dpi=300,bbox_inches='tight',metadata={'Author':'CHONG LIU','Creator':'CHONG LIU'})
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path(__file__).resolve().parents[1]/'results/published_matched_summary.csv');p.add_argument('--output',type=Path,required=True);a=p.parse_args();plot(a.input,a.output)
