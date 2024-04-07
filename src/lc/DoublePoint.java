package lc;

import java.util.*;
/**
 * @author xwp
 * @date 2024/4/6
 * @Description
 */
public class DoublePoint {
    public static void main(String[] args) {
        Solution15 solution15 = new Solution15();
        int[] nums = new int[]{1,4,2,-1,-3,-1,2,-3,-1,-6,-4,-2,3,5,3,6};
        List<List<Integer>> lists = solution15.threeSum(nums);
        lists.forEach(
                it -> System.out.println(it.toString())
        );
    }
}


/*
https://leetcode.cn/problems/3sum/description/?envType=study-plan-v2&envId=top-100-liked
 */
class Solution15 {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>();
        List<Integer> r = new ArrayList<>();
        int len = nums.length-1;
        Arrays.sort(nums);
        for(int i = 0; i < nums.length;i++){
            if(i > 0 && nums[i] == nums[i-1]) continue;
            int j = i+1,k = len;
            while(j<k){
                int q = nums[j], p = nums[k];
                int sum = q + p + nums[i];
                if(sum == 0){
                    r.add(nums[i]);
                    r.add(q);
                    r.add(p);
                    ans.add(new ArrayList<>(r));
                    r.clear();
                    while(j<k && nums[j] == nums[j+1]){
                        j++;
                    }
                    while(j<k && nums[k] == nums[k-1]){
                        k--;
                    }
                    j++;
                    k--;
                }else if(sum > 0){
                    k--;
                }else{
                    j++;
                }
            }
        }
        return ans;
    }
}



