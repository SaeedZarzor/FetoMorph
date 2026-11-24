import os, sys
os.environ["QT_QPA_PLATFORM"]="offscreen"; sys.path.insert(0,os.getcwd())
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from functions.measurements_image import _binarise_brain
from helpers.helpers import iter_hull_edge_sulci
from helpers.scalebar_reader import read_scalebar

F="Examples/full_slices/30/sagittal/band_axis0_s18_idx0113_4x_smooth_sharp.png"
img=cv2.imread(F); rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
mm=read_scalebar(img).mm_per_px
bw=_binarise_brain(img, cv2.COLOR_RGB2GRAY)
cnts,_=cv2.findContours(bw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnt=max(cnts,key=cv2.contourArea); pts=cnt[:,0,:].astype(float)
hull=cv2.convexHull(cnt)[:,0,:]
prom=1.0/mm                                   # Sulcus Depth Threshold = 1 mm
edges=list(iter_hull_edge_sulci(cnt,prom))
HL=(351,244),(439,780)
hl=[e for e in edges if e[0]==HL[0] and e[1]==HL[1]][0]

C_CNT="#d62728"; C_HULL="#1f77b4"; C_S="#2ca02c"; C_D="#ff7f0e"
x0,y0,w,h=cv2.boundingRect(cnt); pad=40
fig=plt.figure(figsize=(15,7.6),dpi=170)
axA=fig.add_axes([0.01,0.06,0.42,0.84]); axB=fig.add_axes([0.47,0.06,0.27,0.84])

# ---------- A: overview ----------
axA.imshow(rgb); axA.set_xlim(x0-pad,x0+w+pad); axA.set_ylim(y0+h+pad,y0-pad)
axA.plot(*np.r_[pts,pts[:1]].T,color=C_CNT,lw=1.6,label="Brain contour")
axA.plot(*np.r_[hull,hull[:1]].T,color=C_HULL,lw=2,label="Convex hull")
n=0
for s,e,sul in edges:
    if (s,e)!=HL: axA.plot([s[0],e[0]],[s[1],e[1]],color=C_HULL,lw=.8,alpha=.5)
    for p,d in sul:
        n+=1; axA.scatter(*p,s=26,color=C_S,zorder=5,edgecolor="white",lw=.6)
axA.plot([HL[0][0],HL[1][0]],[HL[0][1],HL[1][1]],color=C_D,lw=3,zorder=4,label="Hull edge shown in detail")
axA.scatter([],[],s=26,color=C_S,edgecolor="white",label="Sulcus (deepest point)")
axA.set_title("A  Sulci on the slice",loc="left",fontsize=13,fontweight="bold")
axA.legend(loc="lower left",fontsize=9,framealpha=.95); axA.axis("off")
# zoom box
a,b=np.array(HL[0],float),np.array(HL[1],float)
i_s=int(np.argmin(np.hypot(*(pts-a).T))); i_e=int(np.argmin(np.hypot(*(pts-b).T)))
idx=np.arange(i_s,i_e+1) if i_e>=i_s else np.r_[np.arange(i_s,len(pts)),np.arange(0,i_e+1)]
seg=pts[idx]
bx0,bx1=min(a[0],b[0])-175,seg[:,0].max()+45; by0,by1=min(a[1],b[1])-35,max(a[1],b[1])+35
axA.add_patch(Rectangle((bx0,by0),bx1-bx0,by1-by0,fill=False,ec=C_D,lw=1.6,ls="--"))

# ---------- B: zoom on one hull edge ----------
axB.imshow(rgb); axB.set_xlim(bx0,bx1); axB.set_ylim(by1,by0)
axB.plot(*seg.T,color=C_CNT,lw=2.2,label="Contour between s and e")
axB.plot([a[0],b[0]],[a[1],b[1]],color=C_HULL,lw=2.6,label="Hull chord  s–e",zorder=3)
for P,lab,off in ((a,"s",(-18,-14)),(b,"e",(-18,14))):
    axB.scatter(*P,s=80,color=C_HULL,zorder=6,edgecolor="white",lw=1.2)
    axB.annotate(lab,P,xytext=off,textcoords="offset points",fontsize=15,fontweight="bold",color=C_HULL,ha="center",va="center")
ab=b-a
for k,(p,d) in enumerate(sorted(hl[2],key=lambda t:t[0][1]),1):
    p=np.array(p,float); t=np.dot(p-a,ab)/np.dot(ab,ab); foot=a+t*ab
    axB.plot([p[0],foot[0]],[p[1],foot[1]],color=C_D,lw=2.4,zorder=4)
    axB.scatter(*p,s=95,color=C_S,zorder=7,edgecolor="white",lw=1.3)
    axB.scatter(*foot,s=34,color=C_D,zorder=7,edgecolor="white",lw=.8)
    # right-angle mark at the foot
    u=ab/np.linalg.norm(ab); v=(p-foot)/np.linalg.norm(p-foot); q=11
    axB.plot([foot[0]+u[0]*q,foot[0]+(u+v)[0]*q,foot[0]+v[0]*q],[foot[1]+u[1]*q,foot[1]+(u+v)[1]*q,foot[1]+v[1]*q],color=C_D,lw=1.2,zorder=5)
    mid=(p+foot)/2
    axB.annotate(f"depth {k}\n{d*mm:.1f} mm",foot,xytext=(-14,0),textcoords="offset points",
                 fontsize=11.5,fontweight="bold",color="black",va="center",ha="right",
                 bbox=dict(boxstyle="round,pad=.3",fc="white",ec=C_D,lw=1.4),zorder=8)
axB.plot([],[],color=C_D,lw=2.4,label="Depth = perpendicular distance\nfrom sulcus point to the chord")
axB.scatter([],[],s=70,color=C_S,edgecolor="white",label="Sulcus point")
axB.set_title("B  Zoom on one hull edge",loc="left",fontsize=13,fontweight="bold")
axB.axis("off")
h_,l_=axB.get_legend_handles_labels()
fig.legend(h_,l_,loc="center left",bbox_to_anchor=(0.755,0.5),fontsize=10.5,frameon=False,labelspacing=1.1)
fig.text(0.755,0.80,"Each sulcus point is projected\nperpendicularly onto the hull chord;\nthe length of that projection\nis the sulcus depth.",fontsize=11,color="#222",va="top")

# connect box to panel B
for corner,(xb,yb) in (("top",(0,1)),("bottom",(0,0))):
    cp=ConnectionPatch(xyA=(bx1,by0 if corner=="top" else by1),coordsA=axA.transData,xyB=(xb,yb),coordsB=axB.transAxes,color=C_D,lw=1,ls="--",alpha=.7)
    fig.add_artist(cp)
fig.text(0.01,0.965,"How sulcus depth is measured on the image",fontsize=16,fontweight="bold",va="center")
fig.text(0.01,0.025,f"Example: {os.path.basename(F)}   |   pixel size {mm:.3f} mm/px (read from the slice scalebar)   |   depth threshold 1 mm: only sulci at least this deep, and this far above neighbouring points, are counted",fontsize=9,color="#444")
out="figures/sulcus_depth_projection.png"; fig.savefig(out,facecolor="white"); print(out)
print([round(d*mm,2) for _,d in hl[2]])
