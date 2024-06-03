import java.util.*;

// 注意类名必须为 Main, 不要有任何 package xxx 信息
public class Main {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        // 注意 hasNext 和 hasNextLine 的区别
        while (in.hasNextInt()) { // 注意 while 处理多个 case
            int n = in.nextInt();
            char[]  wr = in.next().toCharArray();
            Set<Integer> set = new HashSet<>();
            Map<Integer,Set<Integer>> m = new HashMap<>();
            for (int i = 0; i < n-1; i++) {
                int a = in.nextInt();
                Integer integer = a;
                int b = in.nextInt();
                if(m.containsKey()) {

                }
                if(wr[a] == 'R'){
                    set.add(a);
                }
                if(wr[b] == 'R') set.add(b);
            }
            int k = 1;
            for (Integer s : set) {
                k *=s;
            }
            int res = 0;
            for (int i = 1; i <= k; i++) {
                if(k%i == 0) res++;
            }
            System.out.println(res);
        }
    }
}


