package ai.certifai.solution.object_detection.MyObjDetection;

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ColorConversionTransform;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.event.KeyEvent;
import java.io.File;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.CV_8U;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.helper.opencv_core.RGB;

public class MyDetection_TinyYolo {

    private static Frame frame = null;
    private static ComputationGraph model;
    private static final double  learningrate =1e-5, detectionThreshold = 0.5;
    private static final double[][] priorBoxes = {{1, 3}, {2.5, 6}, {3, 4}, {3.5, 8}, {4, 9}};
    private static final int COLOR_BRG2RGB = 4;
    private static final Logger log = LoggerFactory.getLogger(MyDetection_TinyYolo.class);
    private static final int batch_size = 4, seed = 123, numClasses = 3, numChannels = 3;
    private static final int Epochs = 1 , nBoxes = 5;
    private static final double lamdaNoObj = 0.5, lamdaCoord = 5.0;
    private static List<String> myLabels;
    private static final File myModelFile = new File("D:\\My Pciture 2020\\ConputerVision\\Applied Deep Learning\\model\\My_Model_TinyYolo_TL.zip");
    private static final Scalar GREEN = RGB(0, 255.0, 0);
    private static final Scalar RED = RGB(255, 0, 0);
    private static final Scalar BLUE = RGB(0, 0, 255);
    private static final Scalar YELLOW = RGB(255, 255, 0);
    private static final Scalar BLACK = RGB(0,0,0);
    private static String labeltext = null;
    private static final Scalar[] colormap = {GREEN, YELLOW,BLUE};

    public static void main(String[] args) throws Exception {

        //        STEP 1 : Create iterators
        MyDataSetIterator.setup();
        RecordReaderDataSetIterator trainIter = MyDataSetIterator.trainInterator(batch_size);
        RecordReaderDataSetIterator testIter = MyDataSetIterator.testInterator(1);
        myLabels = trainIter.getLabels();

        /*
                STEP 2 : Load trained model from previous execution
         */
        if (myModelFile.exists()){
            Nd4j.getRandom().setSeed(seed);
            log.info("Loading Model ...");
            model = ModelSerializer.restoreComputationGraph(myModelFile);
            System.out.println(model.summary(InputType.convolutional(
                    MyDataSetIterator.yoloheight,MyDataSetIterator.yolowidth,
                    numChannels
            )));
        }
        else {
            Nd4j.getRandom().setSeed(seed);
            INDArray priors = Nd4j.create(priorBoxes);
            /*
                STEP 2.0 : Train the model using Transfer Learning
            */

            log.info("Build Model....");

            /*
                STEP 2.1: Transfer Learning steps - Load TinyYOLO prebuilt model.
            */
            ComputationGraph preTrained = (ComputationGraph) TinyYOLO.builder().build().initPretrained();

            /*
                STEP 2.2: Transfer Learning steps - Model Configurations.
            */
            FineTuneConfiguration fineTuneConf = getFineTuneConfiguration();

            /*
                STEP 2.3: Transfer Learning steps - Modify prebuilt model's architecture
            */
            model = getComputationGraph(preTrained,priors,fineTuneConf);
            System.out.println(model.summary(InputType.convolutional(
                    MyDataSetIterator.yoloheight,MyDataSetIterator.yolowidth,
                    numChannels
            )));

            /*
                STEP 2.4: Training and Save model.
            */
            log.info("Train Model...");
            UIServer server = UIServer.getInstance();
            StatsStorage storage = new InMemoryStatsStorage();
            server.attach(storage);
            model.setListeners(new ScoreIterationListener(1), new StatsListener(storage));

            for (int i = 1;i < Epochs + 1 ; i++) {
                trainIter.reset();
                while(trainIter.hasNext()){
                    model.fit(trainIter.next());
                }
                log.info("*** Completed Epochs {} ***", i);
            }
            ModelSerializer.writeModel(model,myModelFile,true);
            System.out.println("Model Saved");
        }
        /*
               STEP 3: Evaluate the model's accuracy by using the test iterator.
        */
        log.info("Start Evaluation");
        EvaluateModel(testIter);

        /*
             STEP 4: Inference the model and process the webcam stream and make predictions.
        */
        log.info("Start Inference");
        //InferenceModel(0);


    }



    private static ComputationGraph getComputationGraph(ComputationGraph preTrained, INDArray priors, FineTuneConfiguration fineTuneConf) {

        return new TransferLearning.GraphBuilder(preTrained)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("max_pooling2d_4") //the specified layer and below are "frozen"
                .removeVertexKeepConnections("conv2d_9")
                .removeVertexKeepConnections("outputs")
                .addLayer("conv2d_23", new ConvolutionLayer.Builder(1,1)
                        .nIn(1024)
                        .nOut(nBoxes*(numClasses + 5))
                        .stride(1,1)
                        .convolutionMode(ConvolutionMode.Same)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.IDENTITY)
                        .build(),"leaky_re_lu_8")
                .addLayer("outputs",
                        new Yolo2OutputLayer.Builder()
                                .lambdaNoObj(lamdaNoObj)
                                .lambdaCoord(lamdaCoord)
                                .boundingBoxPriors(priors.castTo(DataType.FLOAT))
                                .build(),
                        "conv2d_23")
                .setOutputs("outputs")
                .build();
    }

    private static FineTuneConfiguration getFineTuneConfiguration() {

        return new FineTuneConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(1.0)
                .updater(new Adam.Builder().learningRate(learningrate).build())
                .l2(0.00001)
                .activation(Activation.IDENTITY)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .build();

    }


    private static void InferenceModel(int CameraID) {
        Thread thread = null;
        NativeImageLoader imageLoader = new NativeImageLoader(MyDataSetIterator.yoloheight
        ,MyDataSetIterator.yolowidth
        ,numChannels
        , new ColorConversionTransform(COLOR_BRG2RGB));
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0,1);
        FrameGrabber grabber = null;

        try{
            grabber = FrameGrabber.createDefault(CameraID);
        } catch (FrameGrabber.Exception e) {
            e.printStackTrace();
        }
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

        try{
            grabber.start();
        } catch (FrameGrabber.Exception e){
            e.printStackTrace();
        }

        CanvasFrame canvas = new CanvasFrame("My Object Detection");
        int w = grabber.getImageWidth();
        int h = grabber.getImageHeight();
        canvas.setCanvasSize(w,h);

        while (true){

            try {
                frame = grabber.grab();
            } catch (FrameGrabber.Exception e) {
                e.printStackTrace();
            }

            if (thread == null) {
                thread = new Thread(()->
                {
                    while (frame != null){

                        try{
                            Mat rawImage = new Mat();
                            //Mat inputImage = new Mat();

                            rawImage = converter.convert(frame);
                            //inputImage = converter.convert(frame);

                            //flip(inputImage, rawImage, 1);

                            Mat resizeImage = new Mat();
                            resize(rawImage,resizeImage, new Size(MyDataSetIterator.yolowidth, MyDataSetIterator.yoloheight));
                            INDArray inputImage = imageLoader.asMatrix(resizeImage);

                            scaler.transform(inputImage);
                            INDArray outputs = model.outputSingle(inputImage);
                            org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model.getOutputLayer(0);
                            List<DetectedObject> objs = yout.getPredictedObjects(outputs,detectionThreshold);
                            YoloUtils.nms(objs,0.4);
                            rawImage = drawResult(objs,rawImage,w,h);
                            canvas.showImage(converter.convert(rawImage));

                        } catch (Exception e){
                            throw new RuntimeException(e);
                        }

                    }

                });
                thread.start();
            }
            KeyEvent t = null;

            try{
                t = canvas.waitKey(33);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            if ((t != null) && (t.getKeyCode()  == KeyEvent.VK_Q)){
                break;
            }
        }
        canvas.dispose();
        try{
            grabber.close();
        } catch (FrameGrabber.Exception e){
            e.printStackTrace();
        }


    }

    private static void EvaluateModel(RecordReaderDataSetIterator test) throws InterruptedException {
        NativeImageLoader loader = new NativeImageLoader();
        //CanvasFrame canvas = new CanvasFrame("Valid Test Data");
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model.getOutputLayer(0);
        Mat ConvertMat = new Mat();
        Mat ConvertMat_Big = new Mat();
        double Total=0, totalGood=0;

        while (test.hasNext()){ // && canvas.isVisible()){
            org.nd4j.linalg.dataset.DataSet dataset = test.next();
            INDArray features = dataset.getFeatures();
            INDArray results = model.outputSingle(features);
            List<DetectedObject> objs = yout.getPredictedObjects(results,detectionThreshold);
            YoloUtils.nms(objs,0.4);
            Mat mat = loader.asMat(features);
            mat.convertTo(ConvertMat,CV_8U,255,0);
            int w = mat.cols() * 2;
            int h = mat.rows() * 2;
            Total += 1;
            if (!objs.isEmpty()){
                for (DetectedObject object : objs) {

                    if (object.getPredictedClass() >= 0 ) {

                        totalGood += 1;
                        //System.out.println("Predict Class : " + object.getPredictedClass() + " Condifence: "+object.getConfidence() + " Current Class : " );
                    }
                }

            }

            //resize(ConvertMat,ConvertMat_Big,new Size(w,h));
            //ConvertMat_Big = drawResult(objs, ConvertMat_Big, w, h);
            //canvas.showImage(converter.convert(ConvertMat_Big));
            //canvas.waitKey();
        }
        System.out.println("Total :" + Total + " Good: " + totalGood + " Precision: " + totalGood/Total*100);
       // canvas.dispose();


    }

    private static Mat drawResult(List<DetectedObject> objects, Mat rawImage, int w, int h) {
        Scalar FontColor = BLACK;
        for (DetectedObject object : objects) {
            double[] xy1 = object.getTopLeftXY();
            double[] xy2 = object.getBottomRightXY();
            String label = myLabels.get(object.getPredictedClass());

            int x1 = (int) Math.round(w * xy1[0] / MyDataSetIterator.gridWidth);
            int y1 = (int) Math.round(h * xy1[1] / MyDataSetIterator.gridHeight);
            int x2 = (int) Math.round(w * xy2[0] / MyDataSetIterator.gridWidth);
            int y2 = (int) Math.round(h * xy2[1] / MyDataSetIterator.gridHeight);

            // Bounding Box
            rectangle(rawImage, new Point(x1,y1), new Point(x2,y2), colormap[object.getPredictedClass()],2,0,0);

            // Labeling
            labeltext = label + " " + String.format("%.2f",object.getConfidence()*100) + "%";
            int[] baseline = {0};
            Size textSize = getTextSize(labeltext,FONT_HERSHEY_DUPLEX,1,1,baseline);
            rectangle(rawImage, new Point(x1 + 2, y2 - 2), new Point(x1 + 2 + textSize.get(0), y2 - 2 - textSize.get(1)), colormap[object.getPredictedClass()],FILLED,0,0);
            putText(rawImage, labeltext, new Point(x1+2,y2-2),FONT_HERSHEY_DUPLEX,1,FontColor);
        }

        return rawImage;
    }

}
