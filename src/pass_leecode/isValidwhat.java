package pass_leecode;

import java.util.*;
import java.util.stream.Collectors;

/**
 * @author xwp
 * @date 2023/7/26
 * @Description
 */
public class isValidwhat {
    public static void main(String[] args) {
        List<String> list = new ArrayList<>();
        String a = "aabd";
        String b = "aabag";
        String e = "aabcd";
        String c = "bab";
        String d = "cab";
        list.add(a);
        list.add(b);
        list.add(c);
        list.add(d);
        list.add(e);
        String[] strings = list.toArray(new String[0]);
        Arrays.sort(strings);
        System.out.println(Arrays.toString(strings));

    }
}

class Solution3 {
    public void isValid(String s) {
        Queue<Integer> queue = new PriorityQueue<>();
        queue.add(2);
        queue.add(3);
        queue.add(1);
        queue.add(5);
        queue.add(9);
        for (Integer integer : queue) {
            System.out.println(integer);
        }
    }
}

class Solution4 {
    public int[] topKFrequent(int[] nums, int k) {
        int[] r = new int[k];

        Map<Integer,Integer> map = new HashMap<>();
        for(int i : nums){
            int n = map.getOrDefault(i,0);
            map.put(i,n+1);
        }
        Queue<Map.Entry<Integer,Integer>> queue = new PriorityQueue<>((o1, o2) ->
            o2.getValue() - o1.getValue());

        map.entrySet().forEach(it->{
            queue.add(it);
        });
        for(int i = 0; i < k; i++){
            r[i] =  queue.remove().getKey();
        }



        return r;
    }
}
