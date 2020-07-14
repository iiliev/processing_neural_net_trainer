import java.text.MessageFormat;

boolean running      = true; 
Trainer t;

int counter = 1;
XYPlot plot = new XYPlot();

void setup(){
  size(800,600);
  
  //t = xorTrainer( new NeuralNetwork( new int[]{2,3,1} ));
  //t = xorTrainerM( new NeuralNetwork( new int[]{2,3,1} ));
  //t = carData2AngleTrainer( new NeuralNetwork(new int[]{4,25,14,4}) );
  //t = carData3AngleTrainer( new NeuralNetwork(new int[]{6,25,14,4}) );
  //t = madeUpDataTrainer( new NeuralNetwork( new int[]{2,5, 6 ,5} ) );
  //t = new ArrayTrainer( "training_data", new NeuralNetwork(new int[]{3,5,4,4}) );

  t = new RuleTrainer( 1000, new NeuralNetwork(new int[]{6,12,7,4}) );  //start from scratch

  //running = false; noLoop();   //use to test performance of loaded nets with printPerformance()
  //t = new RuleTrainer( 1000, new NeuralNetwork("last_net") ); //change to whatever file name is needed

  //runAllTests();exit();
} 
 
void draw(){

  plot.addPoint( t.train() );  
  plot.render();
  counter++;
}
 
void keyPressed() {
  if (key == 's'){                                  // 's'    Toggle loop/noLoop
    if( running ){ running = false; draw(); noLoop(); } 
    else         { running =  true;           loop(); }
  } else if( key == 't' ){
    println(String.format("Current binary error: %.2f; Pool size: %d; Target size: %d",t.binaryError(), t.getPoolSize(), t.getTargetSize()));
  } else if( key == 'v' ){
    t.nn.save("last_net");                 //rename and copy/paste to application where needed
  } else if( key == 'l' ){
    t.nn = new NeuralNetwork("last_net");  //set to whatever network file name you are loading
  } else if( key == 'p' ){
    t.printPerformance();
  }
  
}
