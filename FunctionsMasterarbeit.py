#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('/usr/local/lib/python3.9/site-packages')


# In[2]:


import dionysus as d
import numpy as np
import pandas as pd
import scipy.io
import networkx as nx
import itertools
from itertools import chain
from itertools import compress
import matplotlib.pyplot as plt
import os
import math 
from numpy import linalg as LA
from networkx.generators.random_graphs import erdos_renyi_graph


# In[3]:


def findkcliques(max_cliques, k):
    #give back all subset cliques of order k from a list of maximal cliques
    Allkcliques=[]
    for max_clique in max_cliques:
        Allkcliques.append(list(itertools.combinations(set(max_clique), k)))
    Allkcliques=list(itertools.chain(* Allkcliques))     
    unique=[set(clique) for clique in Allkcliques]
    uniquecliques = [item for index, item in enumerate(unique) if item not in unique[:index]]
    uniquecliques=[tuple(x) for x in uniquecliques]
    return uniquecliques


# In[4]:


def findallcliques(max_cliques,max_dim=100):
    # give back all subset from a list of maximal cliques
    Allkcliques=[]
    for max_clique in max_cliques:
        for k in range(1,min(len(max_clique)+1,max_dim)):
            listofsizek=list(itertools.combinations(set(max_clique), k))
            Allkcliques.append(listofsizek)
    Allkcliques1=list(itertools.chain(* Allkcliques))     
    return list(set( Allkcliques1))


# In[5]:


def ensureallfaces(simplex, dimensions = [2,3]):
    #give back all faces of "dimensions" of a simplex 
    faces=[]
    for dim in dimensions:
            faces.append(list(itertools.combinations(set(simplex), dim)))
    return list(itertools.chain(* faces))


# In[6]:


def getbackinverseVietorisrips(adjma, add_vertex_first = True):
    # compute the barcodes of the inverse Vietoris Rips (2.0)
    f = d.Filtration()
    unique = np.unique(np.array(adjma))
    delta=sorted(unique,reverse = True)
    if add_vertex_first:
        #add all vertices
        for vertex in range(0,len(adjma)):
            f.append(d.Simplex([vertex],1))
    for i in range(0,(len(delta))):
        G = nx.from_numpy_matrix((adjma>=delta[i]), create_using=nx.Graph)
        XX=list(nx.clique.find_cliques(G))
        for sim in XX:
            if len(sim)>1:
                dimensions = range(2,5) if add_vertex_first else range(1,5)
                faces=ensureallfaces(np.asarray(sim), dimensions = dimensions)
                for fa in faces:
                    if i==0:
                        f.append(d.Simplex(np.array(fa),delta[i]))
                    else:
                        f.add(d.Simplex(np.array(fa),delta[i]))
                         
    f.sort(reverse = True)
    m = d.homology_persistence(f)
    dgms = d.init_diagrams(m, f)
    return dgms


# In[7]:


def plotBarcodes(Barcodes,zigzag=False):
    #plot the Barcodes 
    numberofbars=sum([len(Barcodes[dim]) for dim in range(0,len(Barcodes))])
    x =np.arange(0., 10, 0.5) if zigzag else np.arange(0., 1, 0.25)
    t=0.2
    cmap=plt.cm.Dark2
    colors=['tab:blue','tab:orange','tab:purple']
    for dim in range(0,len(Barcodes)):
        for bar in Barcodes[dim]:
            xloc=np.digitize(bar,x)
            xloc=xloc-np.ones([1,2])
            y=(np.ones([1,2])*t)[0]
            plt.plot(bar,y,colors[dim])
            xlimits=[0, 9.5] if zigzag else [0,1]
            plt.xlim(xlimits)
            plt.ylim([0, max(numberofbars*0.2+0.4,4)])
            t=t+0.2


# In[8]:


def getkpowergraphadjma(nodes,allpaths,adjmatrix,k):
    #compute the adjacency matrix of the kth power-graph
    MA=np.zeros([len(nodes),len(nodes)])
    for i in range(0,len(nodes)):
        for j in range(i+1,len(nodes)):
            firstindex=[allpaths[pathnum][0]==i or allpaths[pathnum][0]==j for pathnum in range(0,len(allpaths))]
            secondindex=[allpaths[pathnum][len(allpaths[pathnum])-1]==i or allpaths[pathnum][len(allpaths[pathnum])-1]==j for pathnum in range(0,len(allpaths))]
            exsistapath=np.multiply(np.array(secondindex),np.array(firstindex)) 
            if len(np.where(exsistapath)[0])>0:
                weithspaths=np.zeros([1,len(np.where(exsistapath)[0])])
                for indexpath in np.where(exsistapath)[0]:
                    t=0
                    weigthspath=[adjmatrix[allpaths[indexpath][i],allpaths[indexpath][i+1]] for i in range(0,len(allpaths[indexpath])-1)]
                    weithspaths[t]=np.min(weigthspath)
                    t=t+1
                MA[i,j]=np.min(weithspaths)
                MA[j,i]=MA[i,j]
    return MA


# In[9]:


def getkplex(adjmatrix,k):
    # give back the k-plexes of an graph
    g = nx.from_numpy_matrix(adjmatrix, create_using=nx.Graph)
    maximal_cliques=list(nx.clique.find_cliques(g))
    if k==1:
        return maximal_cliques
    nodes=set(range(0,g.number_of_nodes()))
    kplex=[]
    for clique in maximal_cliques:
        if len(clique)>1:
            kplex.append(set(clique))
            complement=set(nodes)-set(clique)
            neighbors=[]
            for comp in complement:
                newpossibleclique=[]
                for index in clique:
                    newpossibleclique.append(index)
                newpossibleclique.append(comp)
                matrixnewposcli=adjmatrix[:,np.array(newpossibleclique)][np.array(newpossibleclique)]
                degrees=np.array(np.sum(matrixnewposcli>0,axis=0))[0]
                if np.all(degrees>=len(degrees)-k):
                    kplex.append(set(newpossibleclique))
                    neighbors.append(comp)
            t=2
            while len(neighbors)>1:
                dimensions = [t]
                allpossiblecombineighbors=ensureallfaces(neighbors, dimensions)
                neighbors=[]
                for combi in allpossiblecombineighbors:
                    newpossibleclique=[]
                    for index in clique:
                        newpossibleclique.append(index)      
                    for index in combi:
                        newpossibleclique.append(index)
                    matrixnewposcli=adjmatrix[:,np.array(newpossibleclique)][np.array(newpossibleclique)]
                    degrees=np.array(np.sum(matrixnewposcli>0,axis=0))[0]
                    if np.all(degrees>=len(degrees)-k):
                        kplex.append(set(newpossibleclique))
                        for index in combi:
                            neighbors.append(index)
                t=t+1
    kplex=np.unique(kplex) 
    plexes=[len(kpl) for kpl in kplex]
    sortedplexes=sorted(range(len(plexes)), key=lambda k: plexes[k],reverse=True)
    kplex=[tuple(kplex[sort]) for sort in sortedplexes]
    return kplex


# In[10]:


def isakclub(nodes,allpaths,r): 
    #give back if r is a k-club
    complement=nodes-set(r)
    MA=np.zeros([len(r),len(r)])
    ONES=np.ones([len(r),len(r)])
    np.fill_diagonal(ONES,0)
    for index in range(0,len(r)):
        for index1 in range(index+1,len(r)):
            firstindex=[allpaths[pathnum][0]==r[index] or allpaths[pathnum][0]==r[index1] for pathnum in range(0,len(allpaths))]
            secondindex=[allpaths[pathnum][len(allpaths[pathnum])-1]==r[index] or allpaths[pathnum][len(allpaths[pathnum])-1]==r[index1] for pathnum in range(0,len(allpaths))]
            exsistapath=np.multiply(np.array(secondindex),np.array(firstindex))
            trough=[len(set(allpaths[pathnum]).intersection(complement))==0 for pathnum in range(0,len(allpaths))]
            existipaththrouphclub=np.multiply(np.array(exsistapath),np.array(trough))
            if np.any(existipaththrouphclub):
                MA[index,index1]=1
                MA[index1,index]=1
    return np.all(MA==ONES)


# In[11]:


def getallclubs(nodes,allpaths,clans,k):
    #give back all the k-clubs of a list of k-clans
    club=[]
    for clan in clans:
        if isakclub(nodes,allpaths,clan):
            club.append(clan)
    return club


# In[12]:


def transformadjancymatrixkplex(adjancymatrix,maximalcliques,k):
    #transform the adjacency matrix if k-plex
    if k>1:
        for cliques in maximalcliques:
            Matrixcliques=adjancymatrix[:,np.array(cliques)][np.array(cliques)]
            mini=np.amin(Matrixcliques[Matrixcliques>0])
            loc=np.where(Matrixcliques==0)
            loc1=np.where(loc[0]!=loc[1])
            for index in loc1[0]:
                adjancymatrix[:,cliques[loc[0][index]]][cliques[loc[1][index]]]=mini
    return(adjancymatrix)


# In[13]:


def givebackkcliqueMINadjmat(initialadjmatrix,kcliques):
    # give back the adjancency matrix of the k-connectivity graph
    # on the diagonal the weight of the k-cliques
    MA=np.zeros( (len(kcliques), len(kcliques)) )
    k=len(kcliques[0]); 
    for i in range(0,(len(kcliques))):
        Matrixkcliques=initialadjmatrix[:,np.array(kcliques[i])][np.array(kcliques[i])]
        Mini=np.amin(Matrixkcliques[Matrixkcliques>0])
        MA[i,i]=Mini 
        for j in range((i+1),(len(kcliques))):
            Matrixkcliques=initialadjmatrix[:,np.array(kcliques[j])][np.array(kcliques[j])]
            Minj=np.amin(Matrixkcliques[Matrixkcliques>0])
            numberofcommonsimp=len(set(kcliques[i]).intersection(kcliques[j]))
            if numberofcommonsimp==(k-1):
                MA[i,j]=min(Mini,Minj)
                MA[j,i]=min(Mini,Minj)           
    return MA


# In[14]:


def commonface(CC,sim):
    # give back the simplices which form a simplex in the common-face complex
    originalcliques=[CC[index] for index in np.asarray(sim)]
    faces=list(set(findkcliques(originalcliques , len(originalcliques[0])-1)))
    MA=np.zeros((len(faces), len(originalcliques)))
    for i in range(0,len(faces)):
        for j in range(0,len(originalcliques)):
            k=set(faces[i]).intersection(set(originalcliques[j]))
            if len(k)>=(len(originalcliques[0])-1):
                MA[i,j]=1
    MA=MA[MA.sum(axis=1)>1][:]
    simplex=[]
    for i in range(0,len(MA)):
        loc=np.where(MA[i][:])[0]
        simplex.append([sim[index] for index in np.asarray(loc)])
    return(simplex)


# In[15]:


def getbackCCintervals(adjma,Cliques,pthhomo,Commonface):
    #gives back the barcodes of the k-clique connectivity graph 
    f = d.Filtration()
    unique = np.unique(np.array(adjma))
    delta=sorted(unique,reverse = True)
    if len(delta)==1:
        delta.append(0)
    for i in range(0,(len(delta)-1)):
        #adding the vertex
        vertices=np.where(np.diag(adjma)>=delta[i]) if i==0 else np.where(np.bitwise_and(np.diag(adjma)>=delta[i],delta[i-1]>np.diag(adjma))) 
        for vertex in vertices[0]:
            f.append(d.Simplex([vertex],delta[i]))
        # finding cliques in the k-cliques connectivity graph 
        G = nx.from_numpy_matrix((adjma>=delta[i]), create_using=nx.Graph)
        maximalCliques=list(nx.clique.find_cliques(G))
        for clique in maximalCliques:
            if len(clique)>1:
                if Commonface:
                    simplices=commonface(Cliques,clique)
                    for simplex in simplices:
                        faces=ensureallfaces(np.asarray(simplex), dimensions = range(2,pthhomo+3))
                        for fa in faces:
                            f.add(d.Simplex(np.array(fa),delta[i]))
                else:
                    faces=ensureallfaces(np.asarray(clique), dimensions = range(2,pthhomo+3))
                    for fa in faces:
                        f.add(d.Simplex(np.array(fa),delta[i]))
    f.sort(reverse = True)
    m = d.homology_persistence(f)
    dgms = d.init_diagrams(m, f)
    return dgms


# In[16]:


def traductionintervals (dgms,Homology_dimension_max,zigzag=False):
    #transform the death time of our barcodes "inf"->0, 
    # if zigzag "inf"->10
    intervals = [[] for i in range(0, Homology_dimension_max+1)]
    for i, dgm in enumerate(dgms):
        for pt in dgm:
            if i<=Homology_dimension_max:
                if pt.death==float('inf'):
                    death_limit= 10 if zigzag else 0
                    intervals[i].append([pt.birth, death_limit])
                else:
                    intervals[i].append([pt.birth, pt.death])
    return intervals


# In[17]:


def indicator(intervals,delta,zigzag=False):
    # compute the persistence indicator function for the intervals 
    k=len(intervals)
    yy=np.zeros( len(delta))
    for i in range(0,len(delta)):
        if delta[i]==0 and zigzag==False:
            yy[i]=yy[i-1]
        else:
            birth=[xx[0] for xx in intervals]
            death=[xx[1] for xx in intervals]
            birthnum=sum(birth<=delta[i]) if zigzag else sum(birth>=delta[i])
            deathnum=sum(death<=delta[i]) if zigzag else sum(death>=delta[i])
            yy[i]=birthnum-deathnum;
    return yy


# In[18]:


def getcontinusweightmax(nbins,indi,newdelta,linSpace,dim,plot,kk,integral=False):
    # transform persistence indicator into discretize version
    yk=np.digitize(newdelta,linSpace)
    yk[yk>nbins]=nbins
    thebins=np.array(sorted(list(dict.fromkeys(yk)),reverse = True))
    for bb in range(0,len(thebins)):
        index=np.where(yk==thebins[bb])
        index=index[0]
        if thebins[bb]==nbins:
            if integral:
                maax=sum(np.multiply(newdelta[min(index):max(index)]-newdelta[min(index)+1:max(index)+1],indi[min(index)+1:max(index)+1]))
            else:
                maax=np.max(indi[index])
            plot[(kk-2),thebins[bb]-1]=maax
        else:
            if integral:
                maax=sum(np.multiply(newdelta[min(index)-1:max(index)]-newdelta[min(index):max(index)+1],indi[index]))
            else:
                maax=np.max(indi[index])
                maax=max(maax,indi[min(index)-1])
        plot[(kk-2),thebins[bb]-1]=maax
    return plot


# In[19]:


def getplots(adjmatrix,maxdim,nbins,pthhomo=1,Commonface=False,kplex=False,kclub=False,k=0):
    #compute the plots 
    G = nx.from_numpy_matrix(adjmatrix, create_using=nx.Graph)
    if kplex:
        maximalcliques=list(getkplex(adjmatrix,k))
        adjmatrix=transformadjancymatrixkplex(adjmatrix,maximalcliques,k)
    elif kclub:
        allpaths = []
        nodes=set(range(0,G.number_of_nodes()))
        for node in G:
            for n in range(1,k+1):
                allpaths.extend(findPaths(G,node,n))
        adjmatrix=getkpowergraphadjma(nodes,allpaths,adjmatrix,k)
        gpower = nx.from_numpy_matrix(adjmatrix, create_using=nx.Graph)
        maximalcliques=list(nx.clique.find_cliques(gpower)) 
    else :
        maximalcliques=list(nx.clique.find_cliques(G))
    PLOT = [[] for i in range(0, pthhomo+1)]
    for ii in range(0,pthhomo+1):
        PLOT[ii]=np.zeros((maxdim-1,nbins)) 
    maax=max([len(xx) for xx in maximalcliques])
    for cliquesize in range(2,min(maax,maxdim)+1):
        kcliques=findkcliques(maximalcliques,cliquesize)
        if kclub:
            kcliques=getallclubs(nodes,allpaths,kcliques,k)
        if not kcliques:
            continue
        #get adj. matrix of kk-clique connectivity graph
        kcliqueconnecma=givebackkcliqueMINadjmat(adjmatrix,kcliques)
        ## get Barcodes
        dgms=getbackCCintervals(kcliqueconnecma,kcliques,pthhomo,Commonface)
        ##transform Barcodes for plot 
        unique = np.unique(kcliqueconnecma)
        delta=sorted(unique,reverse = True)
        intervals=traductionintervals(dgms,pthhomo)
        linSpace=np.linspace(0,1,(nbins+1))
        newdelta=np.concatenate((linSpace,delta)) 
        newdelta=np.unique(newdelta)
        newdelta=sorted(newdelta,reverse = True)
        newdelta=np.array(newdelta)
        for pp in range(0,pthhomo+1):
            #getindicator function of intervals
            indi=indicator(intervals[pp],newdelta)
            #get plot matrix
            PLOT[pp]=getcontinusweightmax(nbins,indi,newdelta,linSpace,maxdim,PLOT[pp],cliquesize)
    return(PLOT)


# In[20]:


def centralma(n,central=True):
    # simulate a initial graph for Barabasi with one central node or not
    t=0.7 if central else 0.3
    MA=(np.random.rand(n,n)>t)*1
    MA[0,:]=1;
    MA=np.triu(MA)+np.transpose(np.triu(MA))
    np.fill_diagonal(MA, 0)
    return(MA)


# In[21]:


def BarabasiAlbert(n,degree,seed):
    # create a Barabasi-Albert graphs with n nodes, degree, and initial matrix=seed
    MA=np.zeros((n,n))
    MA[0:len(seed),0:len(seed)]=seed;
    for k in range(0,n-len(seed)):
        prob=MA.sum(axis=1)/sum(MA.sum(axis=1))
        p=np.random.choice(range(0,n),degree,replace=False, p=prob)
        MA[k+len(seed),p]=1
        MA[p,k+len(seed)]=1
    MA=np.multiply(MA,np.random.rand(n,n))
    MA=np.triu(MA, 1)+ np.transpose(np.triu(MA, 1))
    return(MA)


# In[22]:


def symmetricrandommatrix(n,a,b):
    #creates a full matrix with random numbers btw (a,b)
    MA=np.zeros([n,n])
    for i in range(0,n):
        for j in range(i+1,n):
            MA[i,j]=np.random.uniform(a,b)
    MA=np.triu(MA)+np.transpose(np.triu(MA))
    return MA


# In[23]:


def LtwodistanceMA(A):
    # the distance matrix for plots
    L2=np.zeros((len(A),len(A)))
    for i in range(0,len(A)):
        for j in range(i,len(A)):
            L2[i,j] = sum([LA.norm((A[i][k,:]-A[j][k,:]),2) for k in range(0,(A[0].shape)[0])])
            L2[j,i]=L2[i,j]
    L2=L2/((A[0].shape)[0])
    L2=L2/np.max(L2)
    return L2


# In[24]:


def pairwisewasserstein2(A):
    # the distance matrix for barcodes with 2-Wasserstein-distance
    DistanceMA=np.zeros((len(A),len(A)))
    for i in range(0,len(A)):
        for j in range(i,len(A)):
            DistanceMA[i,j]=d.wasserstein_distance(A[i], A[j],q=2)
            DistanceMA[j,i]=DistanceMA[i,j]
    DistanceMA=DistanceMA/np.max(DistanceMA)
    return DistanceMA


# In[25]:


def convertloctotimes(location):
    #from the location to the birth and death times of the simplices for dionysus
    times=[[] for j in range(0, len(location))]
    for i in range(0,len(location)):
        if location[i]:
            t=1
            if len(location[i])>1:
                decalle=location[i][1:len(location[i])]
                decalle.append(9)
            else:
                decalle=9
            diff=np.array(decalle)-location[i]
            loc=np.where(diff>1)
            loc=loc[0]
            for index in range(0,len(location[i])):
                if (len(times[i])%2)==0:
                    times[i].append(location[i][index])
                if index in loc:
                    times[i].append(location[i][index]+1) 
        if len(times[i])%2!=0:
            times[i].append(10)
    return times


# In[26]:


def getbackzigzagintervals(Cliques):
    #get back zigzag Barcodes of the clique-complex
    Cliques=[findallcliques(Clique,max_dim=5) for Clique in Cliques]
    cliquei=[[] for j in range(0, len(Cliques))]
    test=list(itertools.chain(* Cliques))
    uni=[set(x) for x in test]
    allcliquespresent = [item for index, item in enumerate(uni) if item not in uni[:index]]
    allcliquespresent=[list(x) for x in allcliquespresent]
    where1=[[] for j in range(0, len(allcliquespresent))]
    for jj in range(0,len(Cliques)):
        cliquei[jj]=[set(x) for x in Cliques[jj]]
    for cliquesnumber in  range(0,len(allcliquespresent)):
        for ii in range(0, len(Cliques)):
            if set(allcliquespresent[cliquesnumber]) in cliquei[ii]:
                 where1[cliquesnumber].append(ii)
    f = d.Filtration(allcliquespresent)
    times=convertloctotimes(where1)
    zz, dgms, cells = d.zigzag_homology_persistence(f, times)
    return dgms


# In[27]:


def getthezigzagfiltration(Maximalcliques,cliquesize,num_nodes,common_face):
    #get back the list of simplices with birth and death time of the clique complex of the k-clique community graph
    #get all k-cliques for each t
    kcliques=[findkcliques(Maximalcliques[t],cliquesize) for t in range(0,len(Maximalcliques))]
    # get all existing k-cliques overall
    allkcliques=list(itertools.chain(*kcliques))
    #unique this existing k-cliques
    unique=[set(x) for x in allkcliques]
    allcliquespresentset = [item for index, item in enumerate(unique) if item not in unique[:index]]
    allcliquespresent=[tuple(x) for x in allcliquespresentset]
    
    ##transform kcliques in list of set 
    cliquei=[[] for j in range(0, len(kcliques))]
    for jj in range(0,len(kcliques)):
        cliquei[jj]=[set(x) for x in kcliques[jj]]
    #find for each k-clique where it was present
    where1=[[] for j in range(0, len(allcliquespresent))]
    for cliquesnumber in  range(0,len(allcliquespresent)):
        for ii in range(0, len(kcliques)):
            if set(allcliquespresent[cliquesnumber]) in cliquei[ii]:
                 where1[cliquesnumber].append(ii)
    #compute the k-clique community adjancency matrix
    kcliqueconnecma=givebackkcliqueMINadjmat(np.ones([num_nodes,num_nodes]), allcliquespresent)
    ### find out maximalcliques of kcliqueconnecma
    graphcliqueconne = nx.from_numpy_matrix(kcliqueconnecma, create_using=nx.Graph)
    maximalKcliques=list(nx.clique.find_cliques(graphcliqueconne))
    if common_face:
        theKcliques=[]
        for max_clique in maximalKcliques:
            common_faceCliques=commonface(allcliquespresent,max_clique)
            for clique_common_face in common_faceCliques:
                theKcliques.append(clique_common_face)
    else:
        theKcliques=maximalKcliques
    AllKcliquesposssible=findallcliques(theKcliques,max_dim=4)
    unique=[set(x) for x in AllKcliquesposssible]
    AllKcliquesposssible = [item for index, item in enumerate(unique) if item not in unique[:index]]
    AllKcliques=[]
    whereAllKcliques=[]
    for numclique in range(0,len(AllKcliquesposssible)):
        location=[]
        if AllKcliquesposssible[numclique]:
            for item in AllKcliquesposssible[numclique]:
                location.append(where1[item])
            intersectionloc=list(set(location[0]).intersection(*location))
            if intersectionloc:
                AllKcliques.append(list(AllKcliquesposssible[numclique]))
                whereAllKcliques.append(sorted(intersectionloc))
    return AllKcliques,whereAllKcliques


# In[28]:


def getbackplotszigzag(Maximalcliques,maxdim,num_nodes,common_face,plots_barcodes):
    #get back the plots for the zigzag k-clique connectivity graphs
    t=0
    maax=max([len(xx) for xx in list(itertools.chain(* Maximalcliques)) ])
    plots=[np.zeros([maxdim-1,10]) for jj in range(0,2)]
    for cliquesize in range(2,min(maax,maxdim)+1):
        ### get all the cliques of k-connectivity graphs and location
        [AllKcliques,whereAllKcliques]=getthezigzagfiltration(Maximalcliques,cliquesize,num_nodes,common_face)
        f = d.Filtration(AllKcliques)
        times=convertloctotimes(whereAllKcliques)
        ## get intervelas 
        zz, dgms_dionysus, cells = d.zigzag_homology_persistence(f, times)
        dgms_tradu=traductionintervals(dgms_dionysus,1,zigzag=True)
        if plots_barcodes:
            plt.figure(t)
            t=t+1
            plotBarcodes(dgms_tradu,zigzag=True)
        linSpace=np.linspace(0,10,num=11)
        for pp in range(0,2):
            indi=indicator(dgms_tradu[pp],linSpace,zigzag=True)
            plots[pp]=getcontinusweightmax(10,indi,linSpace,linSpace,maxdim,plots[pp],cliquesize)
    return plots


# In[29]:


def getzigzag(list_adjmatrix,max_dim,kcliquecommu=False,common_face=False,plots_barcodes=False):   
    # from a list of adjmatrix compute zigzag of barcodes or the plot of the k-clique community zigzag
    adj_for_zigzag=[]
    adj_for_zigzag.append(list_adjmatrix[0])
    for num in range(0,len(list_adjmatrix)-1):
        adj_for_zigzag.append(np.bitwise_or(list_adjmatrix[num]>0,list_adjmatrix[num+1]>0))
        adj_for_zigzag.append(list_adjmatrix[num+1])
    Maximalcliques=[]
    for adj in adj_for_zigzag:
        graph = nx.from_numpy_matrix(adj, create_using=nx.Graph)
        maximal_cliques=list(nx.clique.find_cliques(graph))
        Maximalcliques.append(maximal_cliques)
    if kcliquecommu:
        dgms=getbackplotszigzag(Maximalcliques,max_dim,graph.number_of_nodes(),common_face,plots_barcodes)
    else:
        dgms=getbackzigzagintervals(Maximalcliques)
    return dgms


# In[30]:


def getpercentlimit(adjma, percent):
    #find the threshold i such that the matrix has maximal density "percent"
    for i in  np.arange(0,1,0.02):
        num=np.sum(adjma>=i)/(len(adjma)**2)
        if num < percent:
            return i


# In[31]:


def findPaths(G,u,n):
    #find all the paths in G starting in u with maximal length n
    if n==0:
        return [[u]]
    paths = [[u]+path for neighbor in G.neighbors(u) for path in findPaths(G,neighbor,n-1) if u not in path]
    return paths

