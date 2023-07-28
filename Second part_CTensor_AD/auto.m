function [rateArr,trainingScore,norScoreArr,abnorScoreArr,TPR,FPR,Recall,Precision,f1Score,accuracy,detTime,insertTime,flagScore]=auto(DataName,ID,topK,shrinkLevel,allDataI3,trainingSliceNum,numSliceForSketchStorage,perAno,mu,sigma,percentageCDFpara,seq,path)
[allDataI1,allDataI2,~]=size(DataName);

deteionSliceNum=allDataI3-trainingSliceNum;
anomalySliceNum=round(deteionSliceNum*0.5);

tensorSketch=zeros(numSliceForSketchStorage,allDataI1,allDataI2);
[tensorSketch_I1,tensorSketch_I2,tensorSketch_I3]=size(tensorSketch);
indexZero=1;
projectionTensor=zeros(tensorSketch_I2,tensorSketch_I2,tensorSketch_I3);

anomalySlicePos=randperm(deteionSliceNum,anomalySliceNum)+trainingSliceNum;
anomalySlicePos=sort(anomalySlicePos);
abnormalPointNum =round(allDataI1*allDataI2*perAno);
rateArr=zeros(1,allDataI3);
rateArrCount=1;


if(topK>min(tensorSketch_I1,tensorSketch_I2) || shrinkLevel>min(tensorSketch_I1,tensorSketch_I2))
    errID = 'myComponent:inputError';
    msgtext = 'topK or shrinkLevel is  the expected format, please check the parameter above.';
    ME = MException(errID,msgtext);
    throw(ME);
end



if(deteionSliceNum>(allDataI3-trainingSliceNum) ||   anomalySliceNum>deteionSliceNum )
    errID = 'myComponent:inputError';
    msgtext = 'topK is  the expected format, please check the parameter above.';
    ME = MException(errID,msgtext);
    throw(ME);
end

trainingScore=zeros(1,trainingSliceNum-numSliceForSketchStorage);
abnorScoreArr=zeros(1,anomalySliceNum);
norScoreArr=zeros(1,deteionSliceNum-anomalySliceNum);
abnorScoreArrCount=1;
norScoreArrCount=1;

for i=1:trainingSliceNum
    insertSlice=DataName(:,:,i);
    HorizontalSilce=reshape(insertSlice,1,allDataI1,allDataI2);
    if(i>numSliceForSketchStorage)
       [curScore]=detAno(HorizontalSilce,projectionTensor); 
       trainingScore(1,i-numSliceForSketchStorage)=curScore;
    end
    [tensorSketch,indexZero,projectionTensor,rate]=insertByHorizontalSilce(tensorSketch,tensorSketch_I1,insertSlice,shrinkLevel,indexZero,topK,projectionTensor);   
    rateArr(1,rateArrCount)=rate;
    rateArrCount=rateArrCount+1;
end
trainingScore=sort(trainingScore);
flagScore=trainingScore(1,round((trainingSliceNum-numSliceForSketchStorage)*percentageCDFpara));



TP=0;
FP=0;

FN=0;
TN=0;

timeArr1=zeros(1,deteionSliceNum);
timeArr1Count=1;
timeArr2=zeros(1,deteionSliceNum);
timeArr2Count=1;

for i=1:deteionSliceNum
    testSlice=DataName(:,:,i+trainingSliceNum);
    if(ismember(trainingSliceNum+i,anomalySlicePos)==1)
        outliers=normrnd(mu,sigma,[1,abnormalPointNum]);
        pos=randperm(allDataI1*allDataI2, abnormalPointNum);
        pos=sort(pos);
        for j=1:abnormalPointNum
            posX1=floor(pos(1,j)./allDataI1)+1;
            posX2=mod(pos(1,j),allDataI2);
            if(posX2==0)
                posX1=posX1-1;
                posX2=allDataI2;
            end
            if(posX1<1 || posX1>23 ||  posX2<1 || posX2>23)
                errID = 'myComponent:inputError';
                msgtext = 'Input does not have the expected format, please check the parameter above.';
                ME = MException(errID,msgtext);
               throw(ME);
            end
            testSlice(posX1,posX2)=testSlice(posX1,posX2)+outliers(1,j);
        end
        HorizontalSilce=reshape(testSlice,1,allDataI1,allDataI2);
        time1=cputime;
        [curScore]=detAno(HorizontalSilce,projectionTensor); 
        time2=cputime; 
        timeArr1(1,timeArr1Count)=time2-time1;
        timeArr1Count=timeArr1Count+1;
        
        abnorScoreArr(1,abnorScoreArrCount)=curScore;
        abnorScoreArrCount=abnorScoreArrCount+1;
        if(curScore>flagScore)
            TP=TP+1;
        else
            FN=FN+1;
            time1=cputime;
            [tensorSketch,indexZero,projectionTensor,rate]=insertByHorizontalSilce(tensorSketch,tensorSketch_I1,testSlice,shrinkLevel,indexZero,topK,projectionTensor);   
            time2=cputime; 
            timeArr2(1,timeArr2Count)=time2-time1;
            timeArr2Count=timeArr2Count+1;
            rateArr(1,rateArrCount)=rate;
            rateArrCount=rateArrCount+1;
        end       
    else
        HorizontalSilce=reshape(testSlice,1,allDataI1,allDataI2);
        
        time1=cputime;
        [curScore]=detAno(HorizontalSilce,projectionTensor); 
        time2=cputime; 
        timeArr1(1,timeArr1Count)=time2-time1;
        timeArr1Count=timeArr1Count+1;
        
        norScoreArr(1,norScoreArrCount)=curScore;
        norScoreArrCount=norScoreArrCount+1;
        if(curScore>flagScore)
            FP=FP+1; 
        else
            TN=TN+1;
            time1=cputime;
            [tensorSketch,indexZero,projectionTensor,rate]=insertByHorizontalSilce(tensorSketch,tensorSketch_I1,testSlice,shrinkLevel,indexZero,topK,projectionTensor);
            time2=cputime;
            timeArr2(1,timeArr2Count)=time2-time1;
            timeArr2Count=timeArr2Count+1;
            
            rateArr(1,rateArrCount)=rate;
            rateArrCount=rateArrCount+1;
        end  
    end
end

detTime=sum(timeArr1);
insertTime=sum(timeArr2);
norScoreArr=sort(norScoreArr);
abnorScoreArr=sort(abnorScoreArr);
TPR=(TP/anomalySliceNum);
FPR=(FP/(deteionSliceNum-anomalySliceNum));

Recall=TP/(TP+FN);
Precision=TP/(TP+FP);
f1Score=2*(Precision*Recall)/(Precision+Recall);
accuracy=(TP+TN)/(TP+FP+FN+TN);
end


