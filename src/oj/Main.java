package oj;

import java.util.Scanner;
import java.util.Arrays;
public class Main{
    public static void main(String[] args){
        Scanner s = new Scanner(System.in);
        int res = 0;
        while(s.hasNext()){
            int n = s.nextInt();
            for(int i = 0; i < n; i++){
                String s1 = s.next();
                if(s1.contains("i")) {
                    if(s1.contains("0i")) res++;
                }else {
                    res++;
                }
            }
        }
        System.out.println(res);
    }
}
