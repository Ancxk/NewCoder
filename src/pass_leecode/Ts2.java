package pass_leecode;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

/**
 * @author xwp
 * @date 2023/7/15
 * @Description
 */
public class Ts2 {
    public static void main(String[] args) {
        BufferedInputStream bufferedInputStream = null;
        try {
             bufferedInputStream = new BufferedInputStream(new FileInputStream("D://tfp.txt"));
            byte[] o = new byte[1024];
            int i;
            while( (i=  bufferedInputStream.read(o)) != -1){
                System.out.println(new String(o));

            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }finally {
            try {
                bufferedInputStream.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }


    }
}

class Solution5 {
   public boolean canJump(int[] nums) {
       Integer[] integers = Arrays.stream(nums).boxed().toArray(Integer[]::new);



       Arrays.sort(integers);
       System.out.println(Arrays.toString(integers));
       return true;
   }
}
