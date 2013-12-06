import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.HashMap;
import java.util.Set;

/**
 * Created with IntelliJ IDEA.
 * User: npbool
 * Date: 12/6/13
 * Time: 6:59 PM
 * To change this template use File | Settings | File Templates.
 */
public class InfluenceMaximization {
    public static HashMap<Integer, String> loadKeywords(String fname){
        try{
            BufferedReader br = new BufferedReader(new FileReader(fname));
            HashMap<Integer, String> map = new HashMap<Integer, String>();
            String line;
            while((line = br.readLine())!=null){
                String[] part = line.split("\t");
                map.put(Integer.parseInt(part[1]), part[0]);
            }
            br.close();
            return map;
        } catch(Exception e){
            e.printStackTrace();
            return null;
        }
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
            HashMap map = loadKeywords(argv[2]);
            if(map==null){
                Log.print("Fail to load keywords");
                return;
            }
            BufferedWriter bw = new BufferedWriter(new FileWriter("result.txt"));
            for(Integer seed : seedSet){
                bw.write(map.get(seed)+"\n");
            }
            bw.close();
        } catch(Exception e){
            e.printStackTrace();
        }
    }
}
