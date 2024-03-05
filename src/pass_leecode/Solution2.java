package pass_leecode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author xwp
 * @date 2023/7/9
 * @Description
 */

public class Solution2 {
    public static void main(String[] args) {

        int[] ints = {3,30};
        String ss = minNumber(ints);
        System.out.println(ss);
    }
        public static String minNumber(int[] nums) {
            Integer[] s =  Arrays.stream(nums).boxed().toArray(Integer[] :: new);
            List<String> r = new ArrayList<>();

            Arrays.sort(s,(x,y)->{
                String a = String.valueOf(x);
                String b = String.valueOf(y);
                return (a+b).compareTo(b+a);
            });
            String res ="";
            for(int i = 0; i < s.length; i++){
                res += (String.valueOf(s[i]));
            }
            return res;


    }
}
