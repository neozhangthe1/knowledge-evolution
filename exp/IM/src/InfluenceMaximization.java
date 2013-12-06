import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Set;

/**
 * Created with IntelliJ IDEA.
 * User: npbool
 * Date: 12/6/13
 * Time: 6:59 PM
 * To change this template use File | Settings | File Templates.
 */
public class InfluenceMaximization {
    public static void readGraph(String filename){

    }
    public static void main(String argv[]){
        Graph graph = new Graph();
        if(!graph.loadFile(argv[1])){
            System.out.println("Fail to load file");
        }
        int K = 100;
        double theta = 0.1;
        Set<Integer> seedSet = graph.influenceMaxization(K, theta);

        try{
            BufferedWriter bw = new BufferedWriter(new FileWriter("result.txt"));
            for(Integer seed : seedSet){
                bw.write(seed.toString()+"\n");
            }
            bw.close();
        } catch(Exception e){
            e.printStackTrace();
        }
    }
}
