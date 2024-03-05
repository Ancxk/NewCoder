package pass_leecode;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.ToIntFunction;
import java.util.stream.Collectors;

/**
 * @author xwp
 * @date 2023/8/6
 * @Description
 */
public class My1 {


    public static void main(String[] args) {
        Integer a = 3, b = 4;
        Set<String> res = new HashSet<>();
        String[] strings = res.toArray(new String[0]);

        Map<List<Integer>,Integer> map = new HashMap<>();
        Integer[] r = new Integer[3];
        int[] ints = Arrays.stream(r).mapToInt(i -> i).toArray();

//        new ThreadPoolExecutor(2,3,10, TimeUnit.SECONDS,new ArrayBlockingQueue<>())
//        ReentrantLock reentrantLock = new ReentrantLock();

//        List<List<Integer>> grid = new ArrayList<>();
//        List<Integer> a = new ArrayList<>();
//
//        ArrayList<Integer> b = new ArrayList<>();
//        ArrayList<Integer> c = new ArrayList<>();
//        a.add(0);
//        a.add(0);
//        a.add(1);
//        b.add(0);
//        b.add(0);
//        b.add(0);
//        c.add(0);
//        c.add(0);
//        c.add(0);
//        grid.add(a);
//        grid.add(b);
//        grid.add(c);
//        Solution7 solution7 = new Solution7();
//        int k = solution7.maximumSafenessFactor(grid);
//        Solution8 solution8 = new Solution8();
//        int[] num = {52, 36, 47, 17, 38, 8, 12, 97, 6, 82, 79, 44, 22, 93, 35, 100, 43, 6, 11, 54, 89, 47, 74, 54, 99, 71, 89, 67, 88, 34, 24};
//        List<Integer> collect = Arrays.stream(num).boxed().collect(Collectors.toList());
//
//        int m =  172;
//        long l = System.currentTimeMillis();
//
//        boolean b = solution8.canSplitArray(collect, m);
//        long l2 = System.currentTimeMillis();
//        System.out.println(b);
//        System.out.println(l2-l);

    }

}


class Solution7 {
    int[][] visit;
    int[][] move = {{0,-1},{0,1},{1,0},{-1,0}};
    int len, res = 0;
    public int maximumSafenessFactor(List<List<Integer>> grid) {
        visit = new int[grid.size()][grid.size()];
        len = grid.size();
        List<int[]> steal = new ArrayList<>();
        for(int i = 0; i < grid.size(); i++){
            for(int j = 0; j <  grid.size(); j++){
                if(grid.get(i).get(j) == 1){
                    int[] ints = new int[2];
                    ints[0] = i;
                    ints[1] = j;
                    steal.add(ints);
                }
            }
        }
        int m = grid.size()*3;
        visit[0][0] = 1;
        for (int[] ints : steal) {
            int p = Math.abs(ints[0]) + Math.abs(ints[1]);
            m = Math.min(p,m);
        }
        dfs(grid,steal,0,0,m);
        return res;

    }
    public void dfs(List<List<Integer>> grid,List<int[]> steal,int i,int j,int min){

        if(i == grid.size()-1 && j == grid.size()-1){
            res = Math.max(res,min);
            return;
        }

        for(int k = 0; k < 4; k++){
            int x = i + move[k][0];
            int y = j + move[k][1];
            if(x >= 0 && x < len && y >= 0 && y < len && visit[x][y] == 0){
                int mm = min;
                for (int[] ints : steal) {
                    int p = Math.abs(x-ints[0]) + Math.abs(y-ints[1]);
                    mm = Math.min(p,mm);
                }
                if(mm == 0) continue;
                visit[x][y] = 1;
                dfs(grid,steal,x,y,mm);
                visit[x][y] = 0;
            }
        }

    }
}


class Solution8 {
    public boolean canSplitArray(List<Integer> nums, int m) {
        if(nums.size() <= 2) return true;
        int sum = 0;
        for(int i : nums){
            sum += i;
        }

        return can(nums,sum,0,nums.size()-1,m);
    }
    public boolean can(List<Integer>nums,int sum, int i, int j,int m){
        if(j-i == 1){
            if(sum >= m) return true;
            else return false;
        }
        int t = sum - nums.get(i);


        if(t >= m  &&  can(nums,t,i+1,j,m))  return true;
        t = sum - nums.get(j);
        if(t >= m && can(nums,t,i,j-1,m)) return true;

        return false;
    }
}
