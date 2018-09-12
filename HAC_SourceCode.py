import math
import numpy as np
import copy
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

import time
start_time = time.time()

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
factory = StemmerFactory()
stemmer = factory.create_stemmer()

datajudul = pd.read_csv('E:/Duty/Data/dataskripsi.csv')
casefold = datajudul.applymap(str.lower)
tokenize = casefold.applymap(str.split)
tokenizelist = tokenize['Data'].tolist()

datakeyword = pd.read_csv('E:/Duty/Data/keyword_list.csv')
cf_keyword = datakeyword.applymap(str.lower)
tkn_keyword = cf_keyword.applymap(str.split)
keywordtolist = tkn_keyword['Key'].tolist()

rangeDoc = 100
hasilPreprocessing = []
for i in range(0, rangeDoc):
    isiDoc = []
    for term in tokenizelist[i]:
        hasilStem = stemmer.stem(term)
        hasilStopword = stopword.remove(hasilStem)
        if hasilStopword:
            isiDoc.append(hasilStopword)
    hasilPreprocessing.append(isiDoc)

list_keyword = []
for i in range(0, rangeDoc):
    isiDoc = []
    for term in keywordtolist[i]:
        hasilStem = stemmer.stem(term)
        isiDoc.append(hasilStem)
    list_keyword.append(isiDoc)

#MENGHITUNG TF
fDoc = []
checkTerm = {}
for x in range(0,rangeDoc):    
    fisiDoc = {}
    freq_temp = {}
    for z in hasilPreprocessing[x]:
        count = freq_temp.get(z,0)
        freq_temp[z] = count + 1        
        fisiDoc[z] = freq_temp[z]
        checkTerm[z] = 1         
    fDoc.append(fisiDoc)
listAllTerm = checkTerm.keys()

#MENGHITUNG BOBOT TF
tfDocAll = []
tfd = {}
freqDocBaru = {}
fDocBaru = []
for x in range (0, rangeDoc):
    tfisid = {}
    fisiDocBaru = {}
    for word in listAllTerm:
        if word in hasilPreprocessing[x]:
            f_temp = fDoc[x]
            tfd[word] = 1 + math.log10(f_temp[word])
            tfisid[word] = tfd[word]
            freqDocBaru[word] = f_temp[word]
            fisiDocBaru[word] = freqDocBaru[word]
        else:
            tfd[word] = 0
            tfisid[word] = tfd[word]
            freqDocBaru[word] = 0
            fisiDocBaru[word] = freqDocBaru[word]
    tfDocAll.append(tfisid)
    fDocBaru.append(fisiDocBaru)                


#MENGHITUNG IDF
df = {}
dfterm = {}
idf = {}
for word in listAllTerm:
    for index in range(0, rangeDoc):
        if word in hasilPreprocessing[index]:
            cn = df.get(word,0)
            df[word] = cn + 1
            dfterm[word] = df[word]
            ndf = float(rangeDoc)/float(df[word])
            idf[word] = math.log10(ndf)


#PEMBOBOTAN TF-IDF
tfidfDoc = []
for i in range(0,rangeDoc):
    valTfIdf = {}
    tflist_temp = tfDocAll[i]
    for termDoc in tflist_temp.keys():
        if termDoc in idf.keys():
            tfidf = idf.get(termDoc) * tflist_temp.get(termDoc)
            valTfIdf[termDoc] = tfidf
    tfidfDoc.append(valTfIdf)


#COSINE SIMILARITY & COSINE DISTANCE

temp_powTfIdf = []
for i in range (0, rangeDoc):
    temp_powWtd = {}
    powWtd = {}
    temp_tfidfDocValue = tfidfDoc[i]
    for j in temp_tfidfDocValue:
        powTfIdf = math.pow(temp_tfidfDocValue[j], 2)
        temp_powWtd[j] = powTfIdf
    temp_powTfIdf.append(temp_powWtd)

temp_DotProdDoc = []
for i in range (0,rangeDoc):
    temp_DotProd = []
    for j in range (0,rangeDoc):
        temp_DotProdTerm = {}
        temp_TfIdf_1 = tfidfDoc[i]
        temp_TfIdf_2 = tfidfDoc[j]
        temp_powTfIdf_1 = temp_powTfIdf[i]
        temp_powTfIdf_2 = temp_powTfIdf[j]
        for term1 in temp_powTfIdf_1:
            for term2 in temp_powTfIdf_2:
                if (term1 == term2): 
                    dotprod = temp_TfIdf_1[term1] * temp_TfIdf_2[term1]
                    temp_DotProdTerm[term1] = dotprod
        temp_DotProd.append(temp_DotProdTerm)
    temp_DotProdDoc.append(temp_DotProd)

CosDisDoc = []
for i in range (0,rangeDoc):
    temp_CosSimDoc = []
    temp_CosDisDoc = []
    for j in range (0,rangeDoc):
        DotProdDoc = temp_DotProdDoc[i][j]
        temp_powTfIdf_1 = temp_powTfIdf[i]
        temp_powTfIdf_2 = temp_powTfIdf[j]
        CosSim = round(sum(DotProdDoc.values()) / math.sqrt(sum(temp_powTfIdf_1.values())*sum(temp_powTfIdf_2.values())),15)
        CosDis = 1-CosSim
        temp_CosDisDoc.append(CosDis)
    CosDisDoc.append(temp_CosDisDoc)

print"Waktu Eksekusi :"
print("%s detik" % round((time.time() - start_time),0))
print ""
def pilihLinkage():
    print "Jumlah Dokumen : ", rangeDoc
    print ""
    print "Pilih Parameter Jarak (Linkage)"
    print "1. Single Linkage"
    print "2. Complete Linkage"
    print "3. Average Linkage"
    print "0. Keluar"
    global pilihlinkage
    pilihlinkage = input("Masukkan pilihan : ")
    print ""
awal = 0    
while awal == 0:    
    pilihLinkage()
    if pilihlinkage == 0:
        break
    P = len(CosDisDoc)
    cluster = []
    CosDis = {}
    clustergrup = []
    for i in range (0, P):
        cluster.append(i)
       
    print "Dokumen :"
    print cluster
    print ""
    gabung = []
    thp = 0
    tahap = {}
    while (len(cluster)!=1):
        if thp == 0:
            print "Tahap : ",thp
            print "Cluster : ",cluster
            tahap[thp]=copy.deepcopy(cluster)
        else:
            print "Tahap : ",thp
            listCosDis = {}
            #SINGLE LINKAGE
            if pilihlinkage == 1:
                for doc1 in  cluster:
                    for doc2 in cluster:
                        if doc1 != doc2:
                            if isinstance(doc1, (list))==False and isinstance(doc2, (list))==False :
                                nilaiCosDis = CosDisDoc[doc1][doc2]
                                listCosDis[doc1,doc2] = nilaiCosDis
                            elif isinstance(doc1, (list))==False and isinstance(doc2, (list))==True :
                                newGrup = {}
                                for doc in doc2:
                                    nilaiSementara = CosDisDoc[doc1][doc]
                                    newGrup[doc1,doc] = nilaiSementara
                                urutanGrup = [(value, key) for key, value in newGrup.items()]
                                ambilDokumen = min(urutanGrup)[1]
                                for doc1_grup in ambilDokumen:
                                    for doc2_grup in ambilDokumen:
                                        if doc1_grup != doc2_grup:
                                            nilaiCosDis = CosDisDoc[doc1_grup][doc2_grup]
                                            listCosDis[doc1_grup,doc2_grup] = nilaiCosDis
                            elif isinstance(doc1, (list))==True and isinstance(doc2, (list))==True :
                                newGrup = {}
                                for doc_1 in doc1:
                                    for doc_2 in doc2:
                                        nilaiSementara = CosDisDoc[doc_1][doc_2]
                                        newGrup[doc_1,doc_2] = nilaiSementara
                                urutanGrup = [(value, key) for key, value in newGrup.items()]
                                ambilDokumen = min(urutanGrup)[1]
                                for doc1_grup in ambilDokumen:
                                    for doc2_grup in ambilDokumen:
                                        if doc1_grup != doc2_grup:
                                            nilaiCosDis = CosDisDoc[doc1_grup][doc2_grup]
                                            listCosDis[doc1_grup,doc2_grup] = nilaiCosDis
            #COMPLETE LINKAGE
            elif pilihlinkage == 2:
                for doc1 in  cluster:
                    for doc2 in cluster:
                        if doc1 != doc2:
                            if isinstance(doc1, (list))==False and isinstance(doc2, (list))==False :
                                nilaiCosDis = CosDisDoc[doc1][doc2]
                                listCosDis[doc1,doc2] = nilaiCosDis
                            elif isinstance(doc1, (list))==False and isinstance(doc2, (list))==True :
                                newGrup = {}
                                for doc in doc2:
                                    nilaiSementara = CosDisDoc[doc1][doc]
                                    newGrup[doc1,doc] = nilaiSementara
                                urutanGrup = [(value, key) for key, value in newGrup.items()]
                                ambilDokumen = max(urutanGrup)[1]
                                for doc1_grup in ambilDokumen:
                                    for doc2_grup in ambilDokumen:
                                        if doc1_grup != doc2_grup:
                                            nilaiCosDis = CosDisDoc[doc1_grup][doc2_grup]
                                            listCosDis[doc1_grup,doc2_grup] = nilaiCosDis
                            elif isinstance(doc1, (list))==True and isinstance(doc2, (list))==True :
                                newGrup = {}
                                for doc_1 in doc1:
                                    for doc_2 in doc2:
                                        nilaiSementara = CosDisDoc[doc_1][doc_2]
                                        newGrup[doc_1,doc_2] = nilaiSementara
                                urutanGrup = [(value, key) for key, value in newGrup.items()]
                                ambilDokumen = max(urutanGrup)[1]
                                for doc1_grup in ambilDokumen:
                                    for doc2_grup in ambilDokumen:
                                        if doc1_grup != doc2_grup:
                                            nilaiCosDis = CosDisDoc[doc1_grup][doc2_grup]
                                            listCosDis[doc1_grup,doc2_grup] = nilaiCosDis
            #AVERAGE LINKAGE
            elif pilihlinkage == 3:
                for doc1 in  cluster:
                    for doc2 in cluster:
                        if doc1 != doc2:
                            if isinstance(doc1, (list))==False and isinstance(doc2, (list))==False :
                                nilaiCosDis = CosDisDoc[doc1][doc2]
                                listCosDis[doc1,doc2] = nilaiCosDis
                            elif isinstance(doc1, (list))==False and isinstance(doc2, (list))==True :
                                averageCosDis = []
                                for doc in doc2:
                                    averageCosDis.append(CosDisDoc[doc1][doc])
                                averageCosDisValue = np.mean(averageCosDis)
                                for doc in doc2:
                                    listCosDis[doc1,doc] = averageCosDisValue
                            elif isinstance(doc1, (list))==True and isinstance(doc2, (list))==True :
                                averageCosDis = []
                                for doc_1 in doc1:
                                    for doc_2 in doc2:
                                        averageCosDis.append(CosDisDoc[doc_1][doc_2])
                                averageCosDisValue = np.mean(averageCosDis) 
                                for doc_1 in doc1:
                                    for doc_2 in doc2:
                                        listCosDis[doc_1,doc_2] = averageCosDisValue
            urutanCosDis = [(value, key) for key, value in listCosDis.items()]
            dokDiambil = min(urutanCosDis)[1]
            #print tempCosDis
            gabunganDokumen = []
            dictemp = {}
            print "Doc digabung :",dokDiambil
            combineSingleton = []
            cek = []
            for dokumen in dokDiambil:
                gabunganDokumen.append(dokumen)
                dictemp[dokumen] = int(dokumen)
                for objek in cluster:
                    if isinstance(objek, (list)) :
                        for doclist in objek:
                            for doctemp in gabunganDokumen:            
                                if doclist==doctemp:
                                    cek.append(doclist)
                    elif isinstance(objek, (list)) == False:
                        if dokumen == objek:
                            combineSingleton.append(dokumen)
            if len(combineSingleton) == 2:
                for x in dokDiambil:
                    cluster.remove(x)
                cluster.append(gabunganDokumen)
            for dokumen in dokDiambil:
                if len(cek)>0:
                    gabungClust = []
                    hapusClust = []
                    temp_cek = []
                    doc = 0
                    cluster_cek = []
                    cekclust = 0         
                    for objek in cluster:
                        objekberupaList = isinstance(objek, (list,))
                        if objekberupaList :
                            for dokumen_padaDokDiambil in dokDiambil:                          
                                if dokumen_padaDokDiambil not in objek:
                                    if dokumen_padaDokDiambil in cluster:
                                        doc = dokumen_padaDokDiambil
                                        for grup in cluster:
                                            if isinstance(grup, (list,)) :
                                                for d in dokDiambil:
                                                    if d in grup and d not in cluster:
                                                        grup.append(dokumen_padaDokDiambil)
                                                        for doc1 in gabungClust:
                                                            if doc1 in grup:
                                                                cekclust+=1
                                        cluster.remove(dokumen_padaDokDiambil)
                                    elif dokumen_padaDokDiambil not in cluster:
                                        for elementCluster in cluster:
                                            if isinstance(elementCluster, (list,)):
                                                if dokumen_padaDokDiambil in elementCluster:
                                                    if dokumen_padaDokDiambil not in temp_cek:
                                                        temp_cek.append(dokumen_padaDokDiambil)                                               
                                                    if len(temp_cek) <= len(gabunganDokumen):
                                                        if cluster.index(elementCluster) not in hapusClust:
                                                            hapusClust.append(cluster.index(elementCluster))                                                  
                                                        for semuaDokumen in elementCluster:
                                                            if semuaDokumen not in gabungClust:
                                                                gabungClust.append(semuaDokumen)
                                                                
                    if len(gabungClust)!=0:
                        cluster.append(gabungClust)
                    urutanHapus = [x for n, x in enumerate(cluster) if x in cluster[:n]]
                    if len(urutanHapus)!=0:
                        for x in urutanHapus:
                            cluster.remove(x)
                    else: 
                        for x in sorted(hapusClust, reverse=True):                
                            del cluster[x]            
            print "Cluster : ",cluster  
            tahap[thp]=copy.deepcopy(cluster)      
        thp+=1
        print "==============================================="
    
    inputStep = input("Masukkan titik potong langkah : ")    
    step = inputStep
    #PELABELAN
    #PELABELAN CLUSTER
    tfCluster = []
    for i in range(0, len(tahap[step])):
        if isinstance(tahap[step][i], (list,)):
            tfClusterList = {}
            for doc in tahap[step][i]:
                for x in fDocBaru[doc]:
                    if x in tfClusterList:
                        tfClusterList[x] = tfClusterList[x] + fDocBaru[doc][x]
                    else:
                        tfClusterList[x] = fDocBaru[doc][x]
            tfCluster.append(tfClusterList)
        elif isinstance(tahap[step][i], (list,))==False:
            tfCluster.append(fDocBaru[tahap[step][i]])
    #Bobot TF CLuster
    WtfCluster = []
    for i in range (0, len(tfCluster)):
        WtfTerm = {}
        for x in tfCluster[i]:
            freq = int(tfCluster[i][x])
            if freq != 0 :
                WtfTerm[x] = 1 + math.log10(freq)
            else:
                WtfTerm[x] = 0
        WtfCluster.append(WtfTerm)
    
    #MENGHITUNG IDF CLUSTER
    tempClus = {}
    dfClus = {}
    dftermClus = {}
    idfClus = {}
    for word in listAllTerm:
        for index in range(0, len(tfCluster)):
            if word in tfCluster[index]:
                if tfCluster[index][word] != 0:
                    cn = dfClus.get(word,0)
                    dfClus[word] = cn + 1
                    dftermClus[word] = dfClus[word]
                    ndfClus = float(len(tfCluster))/float(dfClus[word])
                    idfClus[word] = math.log10(ndfClus)
    
    #PEMBOBOTAN TF-IDF CLUSTER
    tf_idf_clust = []
    for i in range(0,len(tfCluster)):
        valTfIdfClus = {}
        tflist_temp = WtfCluster[i]
        for termDoc in tflist_temp.keys():
            if termDoc in idfClus.keys():
                tfidfClus = idfClus.get(termDoc) * tflist_temp.get(termDoc)
                valTfIdfClus[termDoc] = tfidfClus
        tf_idf_clust.append(valTfIdfClus)
    
    label = {}
    for i in range(0,len(tf_idf_clust)):
        valLabel = []
        sorting = sorted(tf_idf_clust[i], key=tf_idf_clust[i].get, reverse=True)   
        no = 0
        for r in sorting:
            if no > 4 :
                break
            else:
                no = no + 1
                valLabel.append(r)
        label[str(tahap[step][i])] = valLabel
              
    #KEYWORD CLSUTER
    keywordAll = {}
    for i in range(0, len(tahap[step])):
        if isinstance(tahap[step][i], (list,)):
            keywordCluster = []
            for doc in tahap[step][i]:
                for term in list_keyword[doc]:
                    if term not in keywordCluster:
                        keywordCluster.append(term)
            keywordAll[str(tahap[step][i])] = keywordCluster
        elif isinstance(tahap[step][i], (list,))==False:
            keywordCluster = []
            for term in list_keyword[tahap[step][i]]:
                if term not in keywordCluster:
                    keywordCluster.append(term)
            keywordAll[str(tahap[step][i])] = keywordCluster
    #AKURASI LABEL DAN KEYWORD
    labelisKeyword = {}
    for cluster in label:
        evaluasiLabelCluster = []
        for term in label[cluster]:
            if term in keywordAll[cluster]:
                evaluasiLabelCluster.append(term)
        labelisKeyword[cluster] = evaluasiLabelCluster
    
    hasilPrecisionLabel = {}
    for cluster in labelisKeyword:
        a = float(len(labelisKeyword[cluster]))
        b = float(len(label[cluster]))
        precisionLabelTiapCluster = (a / b)
        hasilPrecisionLabel[cluster] = precisionLabelTiapCluster
    
    rataPrecision = np.mean(hasilPrecisionLabel.values())
    
    #Evaluasi
    sum_temp_bi = 0
    bi_sort = []
    testes = []
    bi_sort_doc = []
    sc_cls = {}
    ai_doc = {}
    bi_doc = {}
    sc_doc = {}
    for i in range(0, len(tahap[step])):
        if isinstance(tahap[step][i], (list,)):
            #ai
            for doc in tahap[step][i]:
                ai_bag = []
                for doc_pembanding in tahap[step][i]:
                    if doc != doc_pembanding:
                        ai_bag.append(CosDisDoc[doc][doc_pembanding])
                ai = np.mean(ai_bag)
                ai_doc[doc] = ai  
            #bi
            #print "step tahap",tahap[step][i]
    
    for doc1 in tahap[step]:
        if isinstance(doc1, (list)):
            for docclust1 in doc1:
                bisorttes = []                
                for doc2 in tahap[step]:
                    if isinstance(doc2, (list)):
                        if doc1 != doc2:
                            bi_bag_list = []
                            for docclust2 in doc2:
                                bi_bag_list.append(CosDisDoc[docclust1][docclust2])
                            bi_mean = np.mean(bi_bag_list)
                            bisorttes.append(bi_mean)
                    elif isinstance(doc2, (list))==False:
                        bisorttes.append(CosDisDoc[docclust1][doc2])
                if len(bisorttes)!=0:
                    bi = min(bisorttes)
                    bi_doc[docclust1] = bi
                
    for doc_ai in ai_doc:
        for doc_bi in bi_doc:
            if doc_ai == doc_bi:
                sc_doc[doc_ai] = (bi_doc[doc_bi] - ai_doc[doc_ai]) / max(bi_doc[doc_bi], ai_doc[doc_ai])
                sc_cls[str(doc_ai)] = sc_doc[doc_ai]
                for i in range(0, len(tahap[step])):
                    if isinstance(tahap[step][i], (list,))==False:
                        sc_cls[tahap[step][i]] = 0
    nomorCluster = {}
    for i in range(0,len(tahap[step])):
        nomorCluster[i] = tahap[step][i]
    #SC tiap cluster
    sc_avg_cls = {}
    for i in nomorCluster:
        if isinstance(nomorCluster[i], (list,)):
            avgCls = []
            for doc in nomorCluster[i]:
                if len(sc_cls)!=0:
                    avgCls.append(sc_cls[str(doc)])
                else:
                    avgCls.append(0)
            avgSC = np.mean(avgCls)
            sc_avg_cls[i] = avgSC
        else:
            if len(sc_cls)!=0:
                sc_avg_cls[i] = sc_cls[nomorCluster[i]]
            else:
                sc_avg_cls[i] = 0
    #SC keseluruhan
    if len(sc_cls)!=0:
        sc_all = np.mean(sc_cls.values())
    else:
        sc_all = 0
            
    def menu():
        print "Data : "
        print rangeDoc,"dokumen"
        print ""
        print "1. Tampilkan hasil Preprocessing"
        print "2. Tampilkan bobot Tf-Idf"
        print "3. Tampilkan nilai Cosine Distance"
        print "4. Tampilkan jumlah cluster yang terbentuk"
        print "5. Tampilkan Cluster yang terbentuk dan Label Cluster"
        print "6. Tampilkan Silhouette Coefficient Tiap Dokumen"
        print "7. Tampilkan Silhouette Coefficient Tiap Cluster"
        print "8. Tampilkan Silhouette Coefficient Keseluruhan"
        print "9. Tampilkan Precision Label Tiap Cluster"
        print "10. Tampilkan Precision Keseluruhan"
        print "11. Tampilkan Daftar Judul Skripsi"
        print "0. Kembali"
        global pilih
        print ""
        pilih = input("Masukkan pilihan : ")
    back = 0
    while back == 0:    
        menu()
        if pilih == 1:
            print "HASIL PREPROCESSING :"
            print ""
            for i in range(0, len(hasilPreprocessing)):
                print "DOKUMEN-",i,":",hasilPreprocessing[i]
            print ""
            back = input("Ketik 0 untuk kembali ke menu:")
        elif pilih == 2:
            print "BOBOT TF-IDF :"
            i = input("Masukkan nomor dokumen (0-99) :")
            for term in tfidfDoc[i]:
                print term,":", tfidfDoc[i][term]
            print ""
            back = input("Ketik 0 untuk kembali ke menu:")
        elif pilih == 3:
            print "COSINE DISTANCE :"
            print ""
            for doc1 in range(0,len(CosDisDoc)):
                for doc2 in range(0, len(CosDisDoc)):
                    if doc1 != doc2:
                        print "Cosine Distance Doc",doc1,"Doc",doc2,":"
                        print CosDisDoc[doc1][doc2]
            print ""
            back = input("Ketik 0 untuk kembali ke menu:")
        elif pilih == 4:
            print "JUMLAH CLUSTER :", len(tahap[step])
            print ""
            back = input("Ketik 0 untuk kembali ke menu:")
        elif pilih == 5:
            print "HASIL CLUSTER DAN LABEL : "
            for i in range(0,len(tahap[step])):
                if isinstance(tahap[step][i], (list))==False:
                    print "CLUSTER-",i,"|LABEL :",label[str(tahap[step][i])]
                    print "DOKUMEN :",tahap[step][i]
                    print "Judul Dokumen :"
                    print tahap[step][i],":",datajudul['Data'][i]
                    print "____________________________________________________"
                elif isinstance(tahap[step][i], (list)):
                    print "ClLUSTER-",i,"|LABEL :",label[str(tahap[step][i])]
                    print "DOKUMEN :",tahap[step][i]
                    print "Judul Dokumen :"
                    for doc in tahap[step][i]:
                        print doc,":",datajudul['Data'][doc]
                    print "____________________________________________________"
            #print nomorCluster
            print ""
            back = input("Ketik 0 untuk kembali ke menu:")
        elif pilih == 6:
            print "SILHOUETTE COEFFICIENT TIAP DOKUMEN:"
            for i in nomorCluster:
                if isinstance(nomorCluster[i], (list,)):
                    for doc in nomorCluster[i]:
                        print "Cluster",i,"Dokumen",doc,"Silhouette Coefficient:",sc_cls[str(doc)] if len(sc_cls)!=0 else 0
                elif isinstance(nomorCluster[i], (list,))==False:        
                    print "Cluster",i,"Dokumen",nomorCluster[i],"Silhouette Coefficient:",sc_cls[nomorCluster[i]] if len(sc_cls)!=0 else 0
            print ""
            back = input("Ketik 0 untuk kembali ke menu:")
        elif pilih == 7:
            print "SILHOUETTE COEFFICIENT TIAP CLUSTER:"
            for i in sc_avg_cls:
                print "Cluster ",i,"Dokumen :",nomorCluster[i],"| Silhouette Coefficient :",sc_avg_cls[i]
            print ""
            back = input("Ketik 0 untuk kembali ke menu:")
        elif pilih == 8:
            print "SILHOUETTE COEFFICIENT : ",sc_all
            print ""
            back = input("Ketik 0 untuk kembali ke menu:")
        elif pilih == 9:
            print "PRECISION TIAP CLUSTER : "
            for i in range(0,len(tahap[step])):
                print "CLUSTER-",i
                print "LABEL :",label[str(tahap[step][i])]
                print "KEYWORD :",keywordAll[str(tahap[step][i])]
                print "LABEL = KEYWORD :",labelisKeyword[str(tahap[step][i])]
                print "PRECISION : ",hasilPrecisionLabel[str(tahap[step][i])]
                print "____________________________________________________"
            print ""
            back = input("Ketik 0 untuk kembali ke menu:")
        elif pilih == 10:
            print "PRECISION KESELURUHAN : ",rataPrecision
            print ""
            back = input("Ketik 0 untuk kembali ke menu:")
        elif pilih == 11:
            print "DATA SELURUH JUDUL SKRIPSI: "
            print ""
            for i in range(0, len(datajudul)):
                print i,":",datajudul['Data'][i]
            print ""
            back = input("Ketik 0 untuk kembali ke menu:")
        elif pilih == 0:
            break
        

