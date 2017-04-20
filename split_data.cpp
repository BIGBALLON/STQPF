#include <cstdio>
#include <cstring>
using namespace std;

char str[3500000];

int main(){
	freopen("train.txt", "r", stdin );
	freopen("train_split.txt", "w", stdout);
	for( int i = 1; i <= 9000; ++i ){
		if( fgets(str, 3000000, stdin ) == NULL ) return 0;
		printf("%s", str);
	}
	
	freopen("test_split.txt", "w", stdout);
	for( int i = 1; i <= 1000; ++i ){
		if( fgets(str, 3000000, stdin ) == NULL ) return 0;
		printf("%s", str);
	}
	
	return 0;
}
