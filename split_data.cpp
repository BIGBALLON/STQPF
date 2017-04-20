#include <cstdio>
#include <cstring>
using namespace std;

char str[3500000];

int main(){
	freopen("train.txt", "r", stdin );
	freopen("train_A.txt", "w", stdout);
	for( int i = 1; i <= 5000; ++i ){
		if( fgets(str, 3000000, stdin ) == NULL ) return 0;
		printf("%s", str);
	}
	
	freopen("test_B.txt", "w", stdout);
	for( int i = 1; i <= 5000; ++i ){
		if( fgets(str, 3000000, stdin ) == NULL ) return 0;
		printf("%s", str);
	}
	
	return 0;
}
