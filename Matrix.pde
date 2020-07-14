class Matrix {
  int rows;
  int cols;
  float data[][] = null;

  Matrix( float[] a ){  //create a single column matrix
    rows = a.length;
    cols = 1;
    data = new float[rows][1];
    for( int i = 0; i < a.length; i++ )
      data[i][0] = a[i];
  }
  
  Matrix( float[][] a ){
    rows = a.length;
    cols = a[0].length;
    data = a;
  }

  Matrix(int r, int c) {
    rows = r;
    cols = c;
    data = new float[r][c];
    for (int j = 0; j<rows; j++)
      for (int i = 0; i< cols; i++)
        data[j][i] = 0;
  }

  Matrix (Matrix a){
    rows = a.rows;
    cols = a.cols;
    data = new float[rows][cols];
    for (int j = 0; j<a.rows; j++)
      for (int i = 0; i< a.cols; i++)
        data[j][i] = (a.data[j][i]);
  }

  Matrix( JSONArray json ){
    for( int i = 0; i < json.size(); i++ ){
      JSONArray row = json.getJSONArray(i);
      if( data == null ){
        rows = json.size();
        cols = row.size();
        data = new float[rows][cols]; 
      }
      for( int j = 0; j < row.size(); j++ )
        data[i][j] = row.getFloat(j);
    }
  }
  
  Matrix broadcastAndAdd( Matrix m ){  //m is a single column matrix
    Matrix b = new Matrix( m.rows, cols ); 
    for( int i = 0; i < b.rows; i++ )
      for( int j = 0; j < b.cols; j++ ){
        b.data[i][j] = m.data[i][0];
      }
    add( b );
    return this;
  }
  
  Matrix bin(){
    Matrix m = new Matrix( this ); 
    for( int i = 0; i < m.rows; i++ )
      for( int j = 0; j < m.cols; j++ ){
        m.data[i][j] = data[i][j] >= .5 ? 1 : 0;
      }
    return m;
  }

  Matrix mutate( float rate ){
    Matrix m = new Matrix( this ); 
    for( int i = 0; i < m.rows; i++ )
      for( int j = 0; j < m.cols; j++ )
        if( random(1) < rate ){
          float sign = random(1) > .5 ? 1 : -1;
          m.data[i][j] = data[i][j] * rate * sign;
        }
    return m;
  }

  Matrix combine( Matrix other ){
    Matrix m = new Matrix( this ); 
    for( int i = 0; i < m.rows; i++ )
      for( int j = 0; j < m.cols; j++ ){
        if( random(1) > .5 )
          m.data[i][j] = data[i][j];
        else
          m.data[i][j] = other.data[i][j];
      }
    return m;
  }

  Matrix add(Matrix m){
    for (int i = 0; i<m.rows; i++)
      for (int j = 0; j< m.cols; j++){
        data[i][j] = data[i][j] + m.data[i][j];
        //if( Float.isInfinite(data[i][j]) || Float.isNaN(data[i][j])){data[i][j] = Float.MAX_VALUE;println("add explosion");}
      }        
    return this;
  }
  
  Matrix randomize() {
    for (int j = 0; j<rows; j++)
      for (int i = 0; i< cols; i++)
        data[j][i] = randomGaussian(); //multiply the result by whatever threshold needed
     return this;
  }
  
  Matrix subtract(Matrix a){
    if(rows == a.rows && cols == a.cols){
      for (int i = 0; i<rows; i++) {
        for (int j = 0; j<cols; j++){
          data[i][j] -= a.data[i][j];
          //if( Float.isInfinite(data[i][j]) || Float.isNaN(data[i][j])){data[i][j] = -Float.MAX_VALUE;println("subtract explosion");}
        }
      }
    }else
      throw new RuntimeException( String.format("Subraction error: a.r=%d, b.r=%d, a.c=%d, b.c=%d", rows, a.rows, cols, a.cols));
    return this;
  }
  
  Matrix multiply(float n){
    for (int i = 0; i<rows; i++)
      for (int j = 0; j<cols; j++)
        data[i][j] = data[i][j] * n;
    return this;
  }
  
  void activateSigm(){
   for (int j = 0; j<rows; j++)
      for (int i = 0; i< cols; i++)
        data[j][i] = sigmoid(data[j][i]);    
  }
  
  void activateTanh(){
   for (int j = 0; j<rows; j++)
      for (int i = 0; i< cols; i++)
        data[j][i] = tanh(data[j][i]);    
  }
  
  void activateRelu(){
   for (int j = 0; j<rows; j++)
      for (int i = 0; i< cols; i++)
        data[j][i] = relu(data[j][i]);    
  }
    
  boolean isNaN(){
    for( int i = 0; i < rows; i++ )
      for( int j = 0; j < cols; j++ )
        if( Float.isNaN(data[i][j]))
          return true;
    return false;
  }
    
  boolean isInfinite(){
    for( int i = 0; i < rows; i++ )
      for( int j = 0; j < cols; j++ )
        if( Float.isInfinite(data[i][j]))
          return true;
    return false;
  }
    
  void printM(){
    for (int j = 0; j<rows; j++) {
      print("["+j+"]: | ");
      for (int i = 0; i< cols; i++)
        print(data[j][i] + " | ");
      println();
    } 
  }

  void printS( String name ){ //check dimensions when debugging
    println(String.format("%s: %d X %d",name, rows,cols));
  }
  
  JSONArray toJSON(){
    JSONArray result = new JSONArray();
    for( int i = 0; i < rows; i++ ){
      JSONArray row = new JSONArray();
      for( int j = 0; j < cols; j++ )
        row.setFloat(j,data[i][j]);
      result.setJSONArray(i,row);   
    }
    return result;
  }
}

//Everything bellow should be static methods of Matrix
Matrix multiply(Matrix a, Matrix b) {
  Matrix result = new Matrix(a.rows, b.cols);
  for (int i = 0; i<result.rows; i++) {
    for (int j = 0; j<result.cols; j++) {
      float sum = 0;
      for (int k = 0; k< a.cols; k++) {
        sum += a.data[i][k]*b.data[k][j];
        //if( Float.isInfinite(sum) || Float.isNaN(sum)){sum = Float.MAX_VALUE; println("multiply explosion");}
      }
      result.data[i][j] = sum;
    }
  }
  return result;
}

Matrix elementWiseMultiply(Matrix a, Matrix b) {
  Matrix result = new Matrix(a.rows, a.cols);
  for (int i = 0; i<result.rows; i++)
    for (int j = 0; j<result.cols; j++){
      result.data[i][j] = a.data[i][j] * b.data[i][j];
      //if( Float.isInfinite(result.data[i][j]) || Float.isNaN(result.data[i][j])){result.data[i][j] = Float.MAX_VALUE;println("elementWiseMultiply explosion");}
    }
  return result;
}

Matrix elementWiseDivide(Matrix a, Matrix b) {
  Matrix result = new Matrix(a.rows, a.cols);
  for (int i = 0; i<result.rows; i++)
    for (int j = 0; j<result.cols; j++){
      result.data[i][j] = a.data[i][j] / b.data[i][j];
      if( Float.isInfinite(result.data[i][j]) || Float.isNaN(result.data[i][j])){
        result.data[i][j] = Float.MAX_VALUE;  //this seems to help with vanishing/exploding NN gradients
        //println("elementWiseDivide explosion");
      }
    }
  return result;
}

Matrix add( Matrix a, float n ){
  Matrix response = new Matrix(a);
    for (int i = 0; i<a.rows; i++)
      for (int j = 0; j< a.cols; j++)
        response.data[i][j] = a.data[i][j]+n;
  return response;
}

Matrix transpose(Matrix m){
  Matrix result = new Matrix(m.cols, m.rows);
  for (int i = 0; i<m.rows; i++)
    for (int j = 0; j< m.cols; j++)
      result.data[j][i] = m.data[i][j];
  return result;
}

Matrix rowWiseSum( Matrix a ){
  Matrix response = new Matrix( a.rows, 1 );
  for( int i = 0; i < a.rows; i++ )
    for( int j = 0; j < a.cols; j++ )
    response.data[i][0] += a.data[i][j];
  return response;
}

float cost(Matrix al, Matrix y){   //takes 1 column matrices
  float result = 0;                //this one goes to infinity if one of the parameters is close to zero or something, use RMSE bellow
  if(al.rows == y.rows && al.cols == 1 && y.cols == 1)
    for (int i = 0; i<al.rows; i++){
      result += log(al.data[i][0])*y.data[i][0] + log(1-al.data[i][0])*(1-y.data[i][0]); //here
      println( String.format("%.2f,%.2f,%.2f",result, al.data[i][0], y.data[i][0]));
    } else
    throw new RuntimeException(String.format("Cost error: a.r=%d, b.r=%d, a.c=%d, b.c=%d", al.rows, y.rows, al.cols, y.cols)); 
  println();
  return - result / al.rows;
}

float RMSE(Matrix al, Matrix y){   
  float result = 0;
  if(al.rows == y.rows && al.cols == y.cols)
    for (int i = 0; i<al.rows; i++)
      for (int j = 0; j<al.cols; j++)
        result += sq(al.data[i][j]-y.data[i][j]);
  else
    throw new RuntimeException(String.format("RMSE error: a.r=%d, b.r=%d, a.c=%d, b.c=%d", al.rows, y.rows, al.cols, y.cols)); 
  return sqrt(result);
}

float binError(Matrix a, Matrix b){  //use with binary matrices
  float result = 0;
  for (int i = 0; i<a.rows; i++)
    for (int j = 0; j<a.cols; j++)
      result += a.data[i][j] == b.data[i][j] ? 0 : 1;
  return result;
}

float cost( float[] al, float[] y){
  float result = 0;
  for( int i = 0; i < al.length; i++ )
    result += log(al[i])*y[i] + log(1-al[i])*(1-y[i]);
  return - result / al.length;
}

float sigmoid(float x){
  return 1.0/(1.0 + (float)Math.exp(-x));
}

float tanh(float x){
  return (float)Math.tanh(x);
}

float relu(float x){
  return max( 0, x );
}

Matrix sigmoid( Matrix Z ){
  Matrix A = new Matrix(Z);
  A.activateSigm();
  return A;
}

Matrix sigmoidBackward( Matrix dA, Matrix Z ){
  Matrix s = new Matrix(Z);
  s.activateSigm();
  Matrix dZ = elementWiseMultiply( elementWiseMultiply( s, add( s, -1 ).multiply(-1) ), dA );  //dA*s*(1-s)
    //if( dZ.isInfinite() || dZ.isNaN() ){println("sigmoidBackward dZ");dZ.printM();println("dA");dA.printM();
    //  println("s");s.printM();println("Z");Z.printM();println();throw new RuntimeException();}      
  return dZ;
}

Matrix tanh( Matrix Z ){       //To do: tanhBackward()
  Matrix A = new Matrix(Z);
  A.activateTanh();
  return A;
}

Matrix tanhBackward( Matrix dA, Matrix Z ){
  Matrix dZ = elementWiseMultiply( add(elementWiseMultiply( Z, Z ), -1).multiply(-1), dA );
  return dZ; // (1-Z**2)*dA
}

Matrix relu( Matrix Z ){
  Matrix A = new Matrix(Z);
  A.activateRelu();
  return A;
}

Matrix reluBackward( Matrix dA, Matrix Z ){
  Matrix dZ = new Matrix(dA);
  for( int i = 0; i < dZ.rows; i++ )
    for( int j = 0; j < dZ.cols; j++ )
      if(Z.data[i][j] <= 0) dZ.data[i][j] = 0; 
   return dZ;
}

void saveMat(Matrix a,String name){
      PrintWriter output;
      output = createWriter(name); 
      
      output.print(a.rows + " " + a.cols);
      output.println();
      
      for (int i = 0; i < a.rows; i++) {
          for (int j = 0; j < a.cols; j++) {
            output.print(a.data[i][j] + " ");
         }
         output.println();
      }
      
      output.flush();
      output.close();
    //println("saved");
}

Matrix loadMat(String name){
  Matrix mat;
  
  BufferedReader reader;
  reader = createReader(name);
  
  String line;
  int cols_, rows_;
  try{
  line = reader.readLine();
  } catch(IOException e){
    e.printStackTrace();
    line = null;
  } 
  
  
  String pieces[] = split(line, " ");
  rows_ = Integer.parseInt(pieces[0]);
  cols_ = Integer.parseInt(pieces[1]);
  
  mat = new Matrix(rows_,cols_);
  
  for(int i = 0; i< rows_; i++){
    try{
    line = reader.readLine();
    } catch(IOException e){
      e.printStackTrace();
      line = null;
    }
    String piecesInLine[] = split(line, " ");
    
      for(int j = 0; j< cols_; j++){
        mat.data[i][j] = parseFloat(piecesInLine[j]);   
      }
    
  }
  
  println("loaded");
  return mat;
}

void printArr( float[] arr ){
  for( int i = 0; i < arr.length; i++ )
    print( " | " + arr[i] );
  println();
}
