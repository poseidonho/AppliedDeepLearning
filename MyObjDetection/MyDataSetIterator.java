package ai.certifai.solution.object_detection.MyObjDetection;

import ai.certifai.utilities.VocLabelProvider;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

public class MyDataSetIterator {
    private static String loadDir;
    private static Path trainDir,testDir;
    private static FileSplit trainData,testData;
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(MyDataSetIterator.class);
    private static final int seed = 123;
    private static Random rng = new Random(seed);
    private static final int nChannels = 3;
    public static final int gridWidth = 13;
    public static final int gridHeight = 13;
    public static final int yolowidth = 416;
    public static final int yoloheight = 416;

    private static RecordReaderDataSetIterator makeInterator(InputSplit Input, Path dir, int batchSize) throws Exception {

        ObjectDetectionRecordReader recordReader = new ObjectDetectionRecordReader(yoloheight,yolowidth,nChannels
        ,gridHeight,gridWidth, new VocLabelProvider(dir.toString()));

        recordReader.initialize(Input);
        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(recordReader,batchSize,1,1,true);
        iter.setPreProcessor(new ImagePreProcessingScaler(0,1));
        return iter;
    }

    public static RecordReaderDataSetIterator trainInterator(int batchSize) throws Exception {

        return makeInterator(trainData,trainDir, batchSize);
    }

    public static RecordReaderDataSetIterator testInterator(int batchSize) throws Exception {

        return makeInterator(testData,testDir, batchSize);
    }


    public static void setup() throws IOException{
        log.info("Load data");
        LoadData();
        log.info("Load Path:", loadDir);
        System.out.println(loadDir);
        trainDir = Paths.get(loadDir,"train");
        testDir = Paths.get(loadDir,"test");
        trainData = new FileSplit(new File(trainDir.toString()), NativeImageLoader.ALLOWED_FORMATS,rng);
        testData = new FileSplit(new File(testDir.toString()),NativeImageLoader.ALLOWED_FORMATS,rng);
        log.info("Loading Data Complete");

    }

    public static void LoadData() throws IOException {

        loadDir= Paths.get("D:\\My Pciture 2020\\ConputerVision\\Applied Deep Learning\\datasets\\mydata\\images").toString();


    }
}
