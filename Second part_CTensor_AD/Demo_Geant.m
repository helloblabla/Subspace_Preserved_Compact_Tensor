clear;
path='GeantDataNorm.mat ';
load(path);
[allDataI1,allDataI2,~]=size(DataName);
allDataI3=10000;
perAno=0.01;
trainingSliceNum=800;
numSliceForSketchStorageArr=[30];
[~,numSliceForSketchStorageArrLength]=size(numSliceForSketchStorageArr);

percentageCDFparaArr=[1];
[~,percentageCDFparaArrLength]=size(percentageCDFparaArr);
topKArr=[10];
[~,topKArrLength]=size(topKArr);
shrinkLevelArr=[10];
[~,shrinkLevelArrLength]=size(shrinkLevelArr);
numEx=1;
muArr=[0.0001];
[~,muArrLength]=size(muArr);
sigmaArr=[0.00001];
[~,sigmaArrLength]=size(sigmaArr);

for f=1:numSliceForSketchStorageArrLength
    numSliceForSketchStorage=numSliceForSketchStorageArr(1,f);
    for i=1:shrinkLevelArrLength
        shrinkLevel=shrinkLevelArr(1,i);
        for j=1:percentageCDFparaArrLength
            percentageCDFpara=percentageCDFparaArr(1,j);
            for l=1:topKArrLength
                topK=topKArr(1,l);
                ID=randperm(10000000,1);
                for o=1:sigmaArrLength
                    sigma=sigmaArr(1,o);
                    for u=1:muArrLength
                        mu=muArr(1,u);
                        TPRArr=zeros(1,numEx);
                        FPRArr=zeros(1,numEx);
                        RecallArr=zeros(1,numEx);
                        PrecisionArr=zeros(1,numEx);
                        f1ScoreArr=zeros(1,numEx);
                        accuracyArr=zeros(1,numEx);
                        detTimeArr=zeros(1,numEx);
                        insertTimeArr=zeros(1,numEx);
                        for k=1:numEx
                            [rateArr,trainingScore,norScoreArr,abnorScoreArr,TPR,FPR,Recall,Precision,f1Score,accuracy,detTime,insertTime,flagScore]=auto(DataName,ID,topK,shrinkLevel,allDataI3,trainingSliceNum,numSliceForSketchStorage,perAno,mu,sigma,percentageCDFpara,k,path);
                            TPRArr(1,k)=TPR;
                            FPRArr(1,k)=FPR;
                            RecallArr(1,k)=Recall;
                            PrecisionArr(1,k)=Precision;
                            f1ScoreArr(1,k)=f1Score;
                            accuracyArr(1,k)=accuracy;
                            detTimeArr(1,k)=detTime;
                            insertTimeArr(1,k)=insertTime;
                            rateArr=rateArr(rateArr>0);
                        end
                        averTPR=(sum(TPRArr)/numEx);
                        averFPR=(sum(FPRArr)/numEx);
                        averRecall=(sum(RecallArr)/numEx);
                        averPrecision=(sum(PrecisionArr)/numEx);
                        averF1Score=(sum(f1ScoreArr)/numEx);
                        averAccuracy=(sum(accuracyArr)/numEx);
                        averDetTime=(sum(detTimeArr)/numEx);
                        averInsertTime=(sum(insertTimeArr)/numEx);
                        disp("sigma="+sigma+",mu="+mu+",allDataI3="+allDataI3+",trainingSliceNum="+trainingSliceNum+",numSliceForSketchStorage="+numSliceForSketchStorage+",topK="+topK+",shrinkLevel="+shrinkLevel+",percentageCDFpara="+percentageCDFpara+",TPR="+averTPR+",FPR="+averFPR+",Recall="+averRecall+",Precision="+averPrecision+",F1Score="+averF1Score+",Accuracy="+averAccuracy+",averDetTime="+averDetTime+",averInsertTime="+averInsertTime+",flagScore="+flagScore+",ID="+ID);                       
                    end
                    disp(" ");
                end
            end
        end
    end
end