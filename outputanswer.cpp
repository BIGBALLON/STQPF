#include <cstdio>
#include <cstring>

int main(){
	double ans;
	freopen("ans.txt","r",stdin);
	freopen("ans_output.txt","w",stdout);
  for( int i = 0; i < 2000; ++i ){
    scanf( "[ %lf]\n", &ans );
		printf( "%f\n", ans );
	}	
	return 0;
}
