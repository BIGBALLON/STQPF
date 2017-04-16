#include <cstdio>
#include <cstring>

using namespace std;
int main(){
	
	freopen("data_sample.txt","r",stdin);
	freopen("data_sample_c.txt","w",stdout);
	char str[1000000];
	char s1[10], s2[10];
	int x;
	while( scanf( "%s", s1) != EOF ){
   printf("%s",s1);
		for( int i = 0; i < 101*101*60-1; ++i ){
			scanf("%d",&x);
			printf( ",%d",x);
		}
		break;
	}
	return 0;
	
	
}
