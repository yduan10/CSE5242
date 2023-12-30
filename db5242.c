/*
  CSE 5242 Project 2, Fall 2023

  See class project handout for more extensive documentation.
*/

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <asm/unistd.h>
#include <immintrin.h>

/* uncomment out the following DEBUG line for debug info, for experiment comment the DEBUG line  */
//#define DEBUG

/* compare two int64_t values - for use with qsort */
static int compare(const void *p1, const void *p2)
{
  int a,b;
  a = *(int64_t *)p1;
  b = *(int64_t *)p2;
  if (a<b) return -1;
  if (a==b) return 0;
  return 1;
}

/* initialize searches and data - data is sorted and searches is a random permutation of data */
int init(int64_t* data, int64_t* searches, int count)
{
  for(int64_t i=0; i<count; i++){
    searches[i] = random();
    data[i] = searches[i]+1;
  }
  qsort(data,count,sizeof(int64_t),compare);
}

/* initialize outer probes of band join */
int band_init(int64_t* outer, int64_t size)
{
  for(int64_t i=0; i<size; i++){
    outer[i] = random();
  }
}

inline int64_t simple_binary_search(int64_t* data, int64_t size, int64_t target)
{
  int64_t left=0;
  int64_t right=size;
  int64_t mid;

  while(left<=right) {
    mid = (left + right)/2;   /* ignore possibility of overflow of left+right */
    if (data[mid]==target) return mid;
    if (data[mid]<target) left=mid+1;
    else right = mid-1;
  }
  return -1; /* no match */
}

inline int64_t low_bin_search(int64_t* data, int64_t size, int64_t target)
{
  /* this binary search variant
     (a) does only one comparison in the inner loop
     (b) doesn't require an exact match; instead it returns the index of the first key >= the search key.
         That's good in a DB context where we might be doing a range search, and using binary search to
	 identify the first key in the range.
     (c) If the search key is bigger than all keys, it returns size.
  */
  int64_t left=0;
  int64_t right=size;
  int64_t mid;

  while(left<right) {
    mid = (left + right)/2;   /* ignore possibility of overflow of left+right */
    if (data[mid]>=target)
      right=mid;
    else
      left=mid+1;
  }
  return right;
}

inline int64_t low_bin_nb_arithmetic(int64_t* data, int64_t size, int64_t target)
{
  /* this binary search variant
     (a) does no comparisons in the inner loop by using multiplication and addition to convert control dependencies
         to data dependencies
     (b) doesn't require an exact match; instead it returns the index of the first key >= the search key.
         That's good in a DB context where we might be doing a range search, and using binary search to
	 identify the first key in the range.
     (c) If the search key is bigger than all keys, it returns size.
  */
  int64_t left=0;
  int64_t right=size;
  int64_t mid;

  while(left<right) {

    /* YOUR CODE HERE */
    mid = (left + right) / 2;
    right = right - ((right - mid) * (data[mid] >= target));
    left = left + ((mid - left + 1) * (data[mid] < target));
  }
  return right;
}

inline int64_t low_bin_nb_mask(int64_t* data, int64_t size, int64_t target)
{
  /* this binary search variant
     (a) does no comparisons in the inner loop by using bit masking operations to convert control dependencies
         to data dependencies
     (b) doesn't require an exact match; instead it returns the index of the first key >= the search key.
         That's good in a DB context where we might be doing a range search, and using binary search to
	 identify the first key in the range.
     (c) If the search key is bigger than all keys, it returns size.
  */
  int64_t left=0;
  int64_t right=size;
  int64_t mid;
  int64_t mask;

  while(left<right) {

    /* YOUR CODE HERE */
    mid = (left + right) / 2;
    mask = ~((data[mid] >= target) - 1);
    right = (right & ~mask) | (mid & mask);
    left = (left & mask) | ((mid + 1) & ~mask);
  }
  return right;
}

inline void low_bin_nb_4x(int64_t* data, int64_t size, int64_t* targets, int64_t* right)
{
  /* this binary search variant
     (a) does no comparisons in the inner loop by using bit masking operations instead
     (b) doesn't require an exact match; instead it returns the index of the first key >= the search key.
         That's good in a DB context where we might be doing a range search, and using binary search to
	 identify the first key in the range.
     (c) If the search key is bigger than all keys, it returns size.
     (d) does 4 searches at the same time in an interleaved fashion, so that an out-of-order processor
         can make progress even when some instructions are still waiting for their operands to be ready.

     Note that we're using the result array as the "right" elements in the search so no need for a return statement
  */
  int64_t left[4]={0,0,0,0};
  int64_t mid[4];
  right[0]=right[1]=right[2]=right[3]=size;
  int i;

    /* YOUR CODE HERE */
  while ((left[0] < right[0]) || (left[1] < right[1]) || (left[2] < right[2]) || (left[3] < right[3]))
  {
    for (i = 0; i < 4; i++)
    {
      mid[i] = (left[i] + right[i]) / 2;
      int64_t mask = ~((data[mid[i]] >= targets[i])-1);
      right[i] = (mask & mid[i]) | (~mask & right[i]);
      left[i] = ((~mask) & (mid[i] + 1)) | (mask & left[i]);
    }
  }
}


/* The following union type is handy to output the contents of AVX512 data types */
union int8x4 {
  __m256i a;
  int64_t b[4];
};

void printavx(char* name, __m256i v) {
  union int8x4 n;

  n.a=v;
  printf("Value in %s is [%ld %ld %ld %ld ]\n",name,n.b[0],n.b[1],n.b[2],n.b[3]);
}

/*
 * Optinal for using AVX-512

  union int8x8 {
    __m512i a;
    int64_t b[8];
  };

  void printavx512(char* name, __m512i v) {
    union int8x4 n;

    n.a=v;
    printf("Value in %s is [%ld %ld %ld %ld %ld %ld %ld %ld ]\n",name,n.b[0],n.b[1],n.b[2],n.b[3]);
  }

 */


inline void low_bin_nb_simd(int64_t* data, int64_t size, __m256i target, __m256i* result)
{
  /* this binary search variant
     (a) does no comparisons in the inner loop by using bit masking operations instead
     (b) doesn't require an exact match; instead it returns the index of the first key >= the search key.
         That's good in a DB context where we might be doing a range search, and using binary search to
	 identify the first key in the range.
     (c) If the search key is bigger than all keys, it returns size.
     (d) does 4 searches at the same time using AVX2 intrinsics

     See https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-8/intrinsics-for-avx2.html
     for documentation of the AVX512 intrinsics

     Note that we're using the result array as the "right" elements in the search, and that searchkey is being passed
     as an __m256i value rather than via a pointer.
  */

  __m256i aleft = _mm256_set1_epi64x(0);
  __m256i aright = _mm256_set1_epi64x(size);
  __m256i amid;


  __m256i ones = _mm256_set1_epi64x(1);
  __m256i amask;
  __m256i datavec;

 /* YOUR CODE HERE */
  __m256i comparison = _mm256_cmpgt_epi64(aright, aleft);

  while ((int64_t)_mm256_extract_epi64(comparison, 0) != 0 || (int64_t)_mm256_extract_epi64(comparison, 1) != 0 ||
  (int64_t)_mm256_extract_epi64(comparison, 2) != 0 || (int64_t)_mm256_extract_epi64(comparison, 3) != 0)
  {
    amid = _mm256_srli_epi64(_mm256_add_epi64(aright, aleft), 1);
    const long long int *llidata = (const long long int *)data;
    datavec = _mm256_i64gather_epi64(llidata, amid, 8);
        
    amask = _mm256_cmpgt_epi64(datavec, target);
    aright = _mm256_blendv_epi8(aright, amid, amask);
    aleft = _mm256_add_epi64(_mm256_andnot_si256(amask, _mm256_add_epi64(amid, ones)), _mm256_and_si256(amask, aleft));
    comparison = _mm256_cmpgt_epi64(aright, aleft);
  }


  *result = aright;

}

void bulk_bin_search(int64_t* data, int64_t size, int64_t* searchkeys, int64_t numsearches, int64_t* results, int repeats)
{
  for(int j=0; j<repeats; j++) {
    /* Function to test a large number of binary searches

       we might need repeats>1 to make sure the events we're measuring are not dominated by various
       overheads, particularly for small values of size and/or numsearches

       we assume that we want exactly "size" searches, where "size" is the length if the searchkeys array
     */
    for(int64_t i=0;i<numsearches; i++) {
#ifdef DEBUG
      printf("Searching for %ld...\n",searchkeys[i]);
#endif

      // Uncomment one of the following to measure it
      results[i] = low_bin_search(data,size,searchkeys[i]);
      //results[i] = low_bin_nb_arithmetic(data,size,searchkeys[i]);
      //results[i] = low_bin_nb_mask(data,size,searchkeys[i]);

#ifdef DEBUG
      printf("Result is %ld\n",results[i]);
#endif
    }
  }
}

void bulk_bin_search_4x(int64_t* data, int64_t size, int64_t* searchkeys, int64_t numsearches, int64_t* results, int repeats)
{
  register __m256i searchkey_4x;

  for(int j=0; j<repeats; j++) {
    /* Function to test a large number of binary searches using one of the 8x routines

       we might need repeats>1 to make sure the events we're measuring are not dominated by various
       overheads, particularly for small values of size and/or numsearches

       we assume that we want exactly "size" searches, where "size" is the length if the searchkeys array
     */
    int64_t extras = numsearches % 4;
    for(int64_t i=0;i<numsearches-extras; i+=4) {
#ifdef DEBUG
      printf("Searching for %ld %ld %ld %ld  ...\n",
	     searchkeys[i],searchkeys[i+1],searchkeys[i+2],searchkeys[i+3]);
#endif

      // Uncomment one of the following depending on which routine you want to profile

      // Algorithm A
      // low_bin_nb_4x(data,size,&searchkeys[i],&results[i]);

      // Algorithm B
       searchkey_4x = _mm256_loadu_si256((__m256i *)&searchkeys[i]);
       low_bin_nb_simd(data,size,searchkey_4x,(__m256i *)&results[i]);

#ifdef DEBUG
      printf("Result is %ld %ld %ld %ld  ...\n",
	     results[i],results[i+1],results[i+2],results[i+3]);
#endif
    }
    /* a little bit more work if numsearches is not a multiple of 8 */
    for(int64_t i=numsearches-extras;i<numsearches; i++) {

      results[i] = low_bin_nb_mask(data,size,searchkeys[i]);

    }

  }
}


int64_t band_join(int64_t* inner, int64_t inner_size, int64_t* outer, int64_t outer_size, int64_t* inner_results, int64_t* outer_results, int64_t result_size, int64_t bound)
{
  /* In a band join we want matches within a range of values.  If p is the probe value from the outer table, then all
     reccords in the inner table with a key in the range [p-bound,p+bound] inclusive should be part of the result.

     Results are returned via two arrays. outer_results stores the index of the outer table row that matches, and
     inner_results stores the index of the inner table row that matches.  result_size tells you the size of the
     output array that has been allocated. You should make sure that you don't exceed this size.  If there are
     more results than can fit in the result arrays, then return early with just a prefix of the results in the result
     arrays. The return value of the function should be the number of output results.

  */

      /* YOUR CODE HERE */
	int64_t num_results = 0;
	int64_t low_bounds[4];
	int64_t high_bounds[4];
	int64_t start_idxs[4];
	int64_t i, j;
	for (i = 0; i < outer_size-3; i+=4)
	{
		for (j = 0; j < 4 ; j++)
		{
	  		low_bounds[j] = outer[i+j] - bound;
		      	high_bounds[j] = outer[i+j] + bound;
		}
	   	low_bin_nb_4x(inner, inner_size, low_bounds, start_idxs);
		for (j = 0; j < 4 ; j++)
		{
			int64_t k = start_idxs[j];
			while ((k < inner_size) && (inner[k] <= high_bounds[j]))
			{
				inner_results[num_results] = k;
				outer_results[num_results] = i+j;
				num_results++;
				if (num_results == result_size)
				{
					return num_results;
				}
				++k;
			}
		}
	}
	for (; i < outer_size; ++i)
	{
		low_bounds[0] = outer[i] - bound;
		high_bounds[0] = outer[i] + bound;
		int64_t k = low_bin_nb_mask(inner,inner_size,low_bounds[0]);
		while ((k < inner_size) && (inner[k] <= high_bounds[0]))
		{
			inner_results[num_results] = k;
			outer_results[num_results] = i;
			num_results++;
			if (num_results == result_size)
			{
				return num_results;
			}
			++k;
		}
	}	
	return num_results;

}

int64_t band_join_simd(int64_t* inner, int64_t inner_size, int64_t* outer, int64_t outer_size, int64_t* inner_results, int64_t* outer_results, int64_t result_size, int64_t bound)
{
  /* In a band join we want matches within a range of values.  If p is the probe value from the outer table, then all
     reccords in the inner table with a key in the range [p-bound,p+bound] inclusive should be part of the result.

     Results are returned via two arrays. outer_results stores the index of the outer table row that matches, and
     inner_results stores the index of the inner table row that matches.  result_size tells you the size of the
     output array that has been allocated. You should make sure that you don't exceed this size.  If there are
     more results than can fit in the result arrays, then return early with just a prefix of the results in the result
     arrays. The return value of the function should be the number of output results.

     To do the binary search, you could use the low_bin_nb_simd you just implemented to search for the lower bounds in parallel

     Once you've found the lower bounds, do the following for each of the 4 search keys in turn:
        scan along the sorted inner array, generating outputs for each match, and making sure not to exceed the output array bounds.

     This inner scanning code does not have to use SIMD.
  */

      /* YOUR CODE HERE */

	  int64_t num_results = 0;
    __m256i bound_vec = _mm256_set1_epi64x(bound);
    __m256i low_bound_vec, high_bound_vec, start_idx_vec;
    int64_t start_idxs[4];
    int64_t high_bounds[4];
    int64_t i;

    for (i = 0; i <= outer_size - 3; i += 4)
    {
        // Load 4 outer values and compute low and high bounds
        __m256i outer_vals = _mm256_loadu_si256((__m256i*)&outer[i]);
        low_bound_vec = _mm256_sub_epi64(outer_vals, bound_vec);
        high_bound_vec = _mm256_add_epi64(outer_vals, bound_vec);

        // Use SIMD binary search to find starting indexes
        low_bin_nb_simd(inner, inner_size, low_bound_vec, &start_idx_vec);
        _mm256_storeu_si256((__m256i*)start_idxs, start_idx_vec);

        // Store high bounds
        _mm256_storeu_si256((__m256i*)high_bounds, high_bound_vec);

        // Linear scan for each of the 4 start indexes
        for (int64_t j = 0; j < 4; j++)
        {
            int64_t k = start_idxs[j];
            while (k < inner_size && inner[k] <= high_bounds[j])
            {
                if (num_results >= result_size)
                {
                    return num_results;
                }
                inner_results[num_results] = k;
                outer_results[num_results++] = i + j;
                k++;
            }
        }
    }

    // Handle any remaining records with low_bin_nb_mask
    for (; i < outer_size; ++i)
    {
        int64_t low_bound = outer[i] - bound;
        int64_t high_bound = outer[i] + bound;
        int64_t k = low_bin_nb_mask(inner, inner_size, low_bound);
        while (k < inner_size && inner[k] <= high_bound)
        {
            if (num_results >= result_size)
            {
                return num_results;
            }
            inner_results[num_results] = k;
            outer_results[num_results++] = i;
            k++;
        }
    }

    return num_results;
}



int
main(int argc, char *argv[])
{
	 long long counter;
	 int64_t arraysize, outer_size, result_size;
	 int64_t bound;
	 int64_t *data, *queries, *results;
	 int ret;
	 struct timeval before, after;
	 int repeats;
	 int64_t total_results;

	 // band-join arrays
	 int64_t *outer, *outer_results, *inner_results;


	 if (argc >= 5)
	   {
	     arraysize = atoi(argv[1]);
	     outer_size = atoi(argv[2]);
	     result_size = atoi(argv[3]);
	     bound = atoi(argv[4]);
	   }
	 else
	   {
	     fprintf(stderr, "Usage: db5242 inner_size outer_size result_size bound <repeats>\n");
	     exit(EXIT_FAILURE);
	   }

	 if (argc >= 6)
	   repeats = atoi(argv[5]);
	 else
	   {
	     repeats=1;
	   }

	 /* allocate the array and the queries for searching */
	 ret=posix_memalign((void**) &data,64,arraysize*sizeof(int64_t));
	 if (ret)
	 {
	   fprintf(stderr, "Memory allocation error.\n");
	   exit(EXIT_FAILURE);
	 }
	 ret=posix_memalign((void**) &queries,64,arraysize*sizeof(int64_t));
	 if (ret)
	 {
	   fprintf(stderr, "Memory allocation error.\n");
	   exit(EXIT_FAILURE);
	 }
	 ret=posix_memalign((void**) &results,64,arraysize*sizeof(int64_t));
	 if (ret)
	 {
	   fprintf(stderr, "Memory allocation error.\n");
	   exit(EXIT_FAILURE);
	 }

	 /* allocate the outer array and output arrays for band-join */
	 ret=posix_memalign((void**) &outer,64,outer_size*sizeof(int64_t));
	 if (ret)
	 {
	   fprintf(stderr, "Memory allocation error.\n");
	   exit(EXIT_FAILURE);
	 }
	 ret=posix_memalign((void**) &outer_results,64,result_size*sizeof(int64_t));
	 if (ret)
	 {
	   fprintf(stderr, "Memory allocation error.\n");
	   exit(EXIT_FAILURE);
	 }
	 ret=posix_memalign((void**) &inner_results,64,result_size*sizeof(int64_t));
	 if (ret)
	 {
	   fprintf(stderr, "Memory allocation error.\n");
	   exit(EXIT_FAILURE);
	 }


	   /* code to initialize data structures goes here so that it is not included in the timing measurement */
	   init(data,queries,arraysize);
	   band_init(outer,outer_size);

#ifdef DEBUG
	   /* show the arrays */
	   printf("data: ");
	   for(int64_t i=0;i<arraysize;i++) printf("%ld ",data[i]);
	   printf("\n");
	   printf("queries: ");
	   for(int64_t i=0;i<arraysize;i++) printf("%ld ",queries[i]);
	   printf("\n");
	   printf("outer: ");
	   for(int64_t i=0;i<outer_size;i++) printf("%ld ",outer[i]);
	   printf("\n");
#endif

     /* now measure... */

	   /* the code that you want to measure goes here; make a function call */
     gettimeofday(&before,NULL);
	   bulk_bin_search(data,arraysize,queries,arraysize,results, repeats);

	   gettimeofday(&after,NULL);
	   printf("Time in bulk_bin_search loop is %ld microseconds or %f microseconds per search\n", (after.tv_sec-before.tv_sec)*1000000+(after.tv_usec-before.tv_usec), 1.0*((after.tv_sec-before.tv_sec)*1000000+(after.tv_usec-before.tv_usec))/arraysize/repeats);



	  

	   /* the code that you want to measure goes here; make a function call */
     gettimeofday(&before,NULL);
	   bulk_bin_search_4x(data,arraysize,queries,arraysize,results, repeats);

	   gettimeofday(&after,NULL);
	   printf("Time in bulk_bin_search_4x loop is %ld microseconds or %f microseconds per search\n", (after.tv_sec-before.tv_sec)*1000000+(after.tv_usec-before.tv_usec), 1.0*((after.tv_sec-before.tv_sec)*1000000+(after.tv_usec-before.tv_usec))/arraysize/repeats);

	   
	   /* the code that you want to measure goes here; make a function call */
	   total_results=band_join(data, arraysize, outer, outer_size, inner_results, outer_results, result_size, bound);

	   gettimeofday(&after,NULL);
	   printf("Band join result size is %ld with an average of %f matches per output record\n",total_results, 1.0*total_results/(1.0+outer_results[total_results-1]));
	   printf("Time in band_join loop is %ld microseconds or %f microseconds per outer record\n", (after.tv_sec-before.tv_sec)*1000000+(after.tv_usec-before.tv_usec), 1.0*((after.tv_sec-before.tv_sec)*1000000+(after.tv_usec-before.tv_usec))/outer_size);

    /* the code that you want to measure goes here; make a function call */
	   total_results=band_join_simd(data, arraysize, outer, outer_size, inner_results, outer_results, result_size, bound);

	   gettimeofday(&after,NULL);
	   printf("Band join SIMD result size is %ld with an average of %f matches per output record\n",total_results, 1.0*total_results/(1.0+outer_results[total_results-1]));
	   printf("Time in band_join_simd loop is %ld microseconds or %f microseconds per outer record\n", (after.tv_sec-before.tv_sec)*1000000+(after.tv_usec-before.tv_usec), 1.0*((after.tv_sec-before.tv_sec)*1000000+(after.tv_usec-before.tv_usec))/outer_size);


#ifdef DEBUG
	   /* show the band_join results */
	   printf("band_join results: ");
	   for(int64_t i=0;i<total_results;i++) printf("(%ld,%ld) ",outer_results[i],inner_results[i]);
	   printf("\n");
#endif

}

