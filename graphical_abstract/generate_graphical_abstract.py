"""Generate an editable monochrome graphical abstract; no model training.
Run: python generate_graphical_abstract.py
Dependencies: matplotlib, numpy. Values: locally generated fractions/evaluation/metrics.json.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','font.size':12})
fig=plt.figure(figsize=(16,9),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,160),ylim=(0,90));ax.axis('off')
def text(x,y,s,size=14,weight='normal',ha='left',va='center',**kw):return ax.text(x,y,s,fontsize=size,weight=weight,ha=ha,va=va,**kw)
def arrow(x,y,xx,yy):ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=19,lw=1.6,color='black'))
def box(x,y,w,h,label=None,fill='white',size=13):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.45,rounding_size=1',fc=fill,ec='.25',lw=1.2))
 if label:text(x+w/2,y+h/2,label,size,ha='center')
def strain(x,y,label):
 ax.add_patch(Circle((x,y),2.1,fc='.94' if label!='T' else '.23',ec='.2',lw=1.1))
 text(x,y,label,11,ha='center',color='white' if label=='T' else 'black')
text(7,84,'Discover strain representations. Reuse them for later prediction.',23,'bold')
text(7,79,'Microbial coculture responses → CDLD → fixed representation reuse',14,color='.3')
for x,num,title in [(7,'01','Observe at 24 h'),(60,'02','Discover strain representations'),(116,'03','Reuse at 48 h')]:
 text(x,71,num,11,'bold',color='.45');text(x,67,title,16,'bold')
# Observation rows, symbolic response rather than simulated data.
for y,labels in [(58,['A','B','T']),(49,['A','C','T'])]:
 box(7,y-3.5,26,7)
 for x,l in zip([12,20,28],labels):strain(x,y,l)
 arrow(35,y,41,y)
 text(44,y,'Response',12)
text(7,40,'Shared strains across cocultures',12,color='.3')
arrow(52,53,59,53)
box(61,57,43,6,'CDLD discovery',fill='.94',size=14)
# Symbolic vectors; no invented latent measurements.
for y,label,vals in [(51,'A',[.3,.75,.5,.9,.4,.65]),(46,'B',[.8,.4,.65,.3,.9,.5]),(41,'T',[.5,.9,.3,.7,.45,.8])]:
 text(64,y,label,12,'bold')
 for j,v in enumerate(vals):ax.add_patch(Rectangle((69+j*4.5,y-1.45),3.7,2.9,fc=str(v),ec='.2',lw=.45))
text(83,35,'Without prespecified strain features',11,ha='center')
arrow(105,49,114,49)
box(117,47,35,9,'Train a new 48 h\nPredictor',fill='.94',size=14)
text(134.5,62,'48 h observations',13,ha='center');arrow(134.5,59,134.5,56.5)
text(134.5,42,'Strain vectors fixed',12,'bold',ha='center')
arrow(134.5,39,134.5,35)
text(134.5,31,'Predict unseen combinations',12,ha='center')
# A subtle divider separates empirical result from schematic mechanism.
ax.plot([7,153],[27,27],color='.75',lw=.9)
text(7,22,'Earlier observations remain useful',18,'bold')
text(7,17,'Fixed reuse had lower error than de novo discovery\nwith 10%, 25% or 50% of 48 h training observations.',13,va='top',linespacing=1.6)
text(7,6,'Exploratory experiment · Validation set held fixed',10,color='.35')
plot=fig.add_axes([.63,.085,.29,.205])
records=json.loads((OUT.parent/'fractions/evaluation/metrics.json').read_text())
x=np.array([10,25,50,100])
def means(model):
 return np.array([np.mean([r['mse'] for r in records if r['pct']==pct and r['model']==model]) for pct in x])
lf=means('LF');ru=means('RU')
plot.plot(x,lf,'o-',color='black',lw=1.8,ms=5,label='Fixed reuse')
plot.plot(x,ru,'s--',color='.45',lw=1.5,ms=4.5,mfc='white',label='De novo discovery')
plot.set(xlim=(5,105),ylim=(0,.255),xticks=x,yticks=[0,.1,.2])
plot.set_xlabel('48 h training observations (%)',fontsize=10,labelpad=3);plot.set_ylabel('Test MSE',fontsize=10)
plot.tick_params(labelsize=9);plot.spines[['top','right']].set_visible(False)
plot.legend(frameon=False,fontsize=9,loc='upper right',handlelength=2)
plot.text(100,.091,'No established advantage\nat 100%',fontsize=9,ha='right',va='center',color='.25')
plot.annotate('',xy=(100,.048),xytext=(98,.074),arrowprops={'arrowstyle':'-','color':'.4','lw':.8})
# Exact display values supplied as a simple standalone reproducibility input.
np.savetxt(OUT/'learning_curve_values.csv',np.column_stack([x,lf,ru]),delimiter=',',header='training_percent,fixed_reuse_MSE,de_novo_MSE',comments='',fmt=['%d','%.5f','%.5f'])
fig.savefig(OUT/'graphical_abstract.png',dpi=240,facecolor='white')
fig.savefig(OUT/'graphical_abstract.svg',facecolor='white')
plt.close(fig)
