class Trainer{
  NeuralNetwork nn;  

  float         train(){ return 0; }
  float   binaryError(){ return 0; }
  int     getPoolSize(){ return 0; }
  int   getTargetSize(){ return 0; }
  void printPerformance(){ println("nothing"); }

}

class MatrixTrainer extends Trainer{
  Matrix inputs;
  Matrix targets;

  MatrixTrainer( Matrix is, Matrix ts, NeuralNetwork n ){
    inputs  = is;
    targets = ts;
    nn      = n;
    if( inputs.rows != nn.inputSize || targets.rows != nn.outputSize )
      throw new RuntimeException( String.format( "Incompatible data: input data - %d; net.in - %d; target data - %d; net.out - %d", 
                                                  inputs.rows, 
                                                  nn.inputSize, 
                                                  targets.rows, 
                                                  nn.outputSize ));
  }
  
  float train(){
    return nn.train( inputs, targets ) / inputs.cols; /// divide ok???
  }
  
  float binaryError(){
    return binError( nn.predict(inputs).bin(), targets );
  }

  int getPoolSize(){ 
    return inputs.cols; 
  }  
  
  int getTargetSize(){ 
    return targets.rows; 
  }
}

class ArrayTrainer extends Trainer{
  ArrayList<Matrix> inputs;
  ArrayList<Matrix> targets;
  
  ArrayTrainer( ArrayList<Matrix> is, ArrayList<Matrix> ts, NeuralNetwork n ){
    inputs  = is;
    targets = ts;
    nn      = n;
    if( inputs.get(0).rows != nn.inputSize || targets.get(0).rows != nn.outputSize )
      throw new RuntimeException( String.format( "Incompatible data: input data - %d; net.in - %d; target data - %d; net.out - %d", 
                                                  inputs.get(0).rows, 
                                                  nn.inputSize, 
                                                  targets.get(0).rows, 
                                                  nn.outputSize ));
  }

  ArrayTrainer( String fileName, NeuralNetwork n ){
    nn = n;
    MessageFormat loadSavePathFormatter = new MessageFormat("data/{0}.csv");
    inputs  = new ArrayList<Matrix>();
    targets = new ArrayList<Matrix>();
    float[] input  = new float[nn. inputSize];
      float[] target = new float[nn.outputSize];
    Table table = loadTable( loadSavePathFormatter.format(new Object[] {fileName}), "header" );
    for (TableRow row : table.rows()) {
      for( int i = 0; i < nn.inputSize; i++ )
        input[i] = row.getFloat("i"+Integer.toString(i));
      inputs.add(new Matrix(input));

      for( int i = 0; i < nn.outputSize; i++ )
        target[i] = row.getFloat("t"+Integer.toString(i));
      targets.add(new Matrix(target));
    }
  }
  
  float train(){
    float error = 0;
    for( int i = 0; i < inputs.size(); i++ ){
      error += sq(nn.train( inputs.get(i), targets.get(i) )); //squaring, summing and taking root will make it the same as MatrixTrainer
      //println(error);
    }
    return sqrt(error) / inputs.size();
  }
    
  float binaryError(){
    float error = 0;
    for( int i = 0; i < inputs.size(); i++ )
      error += binError( nn.predict( inputs.get(i) ).bin(), targets.get(i) );      
    return error;
  }
  
  int getPoolSize(){ 
    return inputs.size(); 
  }

  int getTargetSize(){ 
    return targets.get(0).rows; 
  }
}

class RuleTrainer extends Trainer{
  int iterations;
  ArrayList<float[]> combinedData;
  
  RuleTrainer( int it, NeuralNetwork n ){
    iterations = it;
    nn         = n;
    //nn.activations = new String[]{"relu","relu","sigmoid"};
    combinedData = applyRule();
  }

  ArrayList<float[]> applyRule(){
    float velThreshold = .06;   //this is 6% of max speed which is 10
    float angThreshold = 0.032; //this is .1 radians mapped to 0 to maxAngle which is Pi
    float[] cmdAcc     = new float[]{1,0,0,0};
    float[] cmdAccRght = new float[]{1,0,1,0};
    float[] cmdNothin  = new float[]{0,0,0,0};
    float[] cmdRght    = new float[]{0,0,1,0};
    ArrayList<float[]> response = new ArrayList<float[]>();
    for( int i = 0; i < iterations; i++ ){
      float[] input = { random(1) > .5 ? random(velThreshold,1) : random( velThreshold ), //give 50/50 chance to numbers bellow threshold 
                        random(1) > .5 ? random(angThreshold,1) : random( angThreshold ), //give 50/50 chance to numbers bellow threshold 
                        random(1), 
                        random(1), 
                        random(1), 
                        random(1) };
      //println(String.format("%.2f, %.2f, %.2f, %.2f, %.2f, %.2f",input[0],input[1],input[2],input[3],input[4],input[5]));
      float[] target;
      if( input[0] < velThreshold )   //velocity
        if( input[1] <= angThreshold ) target = cmdAcc;
        else                           target = cmdAccRght;
      else
        if( input[1] <= angThreshold ) target = cmdNothin;
        else                           target = cmdRght;
      response.add(  input );  //every other item is an input
      response.add( target );  //every other item is a target
    }
    
    return response;
  }

  float train(){
    float error     =    0;
    for( int i = 0; i < combinedData.size(); i+=2 ){
      float[]  input = combinedData.get(i  );  //every other item is an input
      float[] target = combinedData.get(i+1);  //every other item is a target
      error += nn.train( input, target ); 
    }
    return error / iterations;
  }
    
  float binaryError(){
    float error = 0;
    for( int i = 0; i < combinedData.size(); i+=2 ){
      float[]  input = combinedData.get(i  );  //every other item is an input
      float[] target = combinedData.get(i+1);  //every other item is a target
      error += binError( nn.predict( input ).bin(), transpose(new Matrix(target)) ); 
    }
    return error;
  }

  int getPoolSize(){ 
    return iterations; 
  }

  int getTargetSize(){ 
    return combinedData.get(1).length; 
  }

  void printPerformance(){
    println("--------------------------------------------------------------------------------"); println();
    for( int i = 0; i < combinedData.size(); i+=2 ){
      float[]  input = combinedData.get(i  );  //every other item is an input
      float[] target = combinedData.get(i+1);  //every other item is a target
      Matrix     OUT = nn.predict( input );
      float[] output = OUT.data[0];
      float[]    bin = OUT.bin().data[0];
      float binError = binError( OUT.bin(), transpose(new Matrix(target)) ); 
      print("In: "); printArr(  input );
      print("Out:"); printArr( output );
      print("Bin:"); printArr(    bin );
      print("Tgt:"); printArr( target );
      println( "Err: " + binError );
      println();
    }
  }
}
