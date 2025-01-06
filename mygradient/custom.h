
#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884197169399375105820974944
#endif


// declare motion variable type
typedef enum {
    STOP,
    FORWARD,
    LEFT,
    RIGHT
} motion_t;

// declare variables

typedef struct 
{
  // Old variables from example gradient
  uint16_t gradient_value;
  uint16_t recvd_gradient;
  uint8_t new_message;
  message_t msg;

  //variable to keep track of the light
  uint16_t light;
  uint16_t obstacle;
  uint16_t obstacle_nearby;
  uint16_t obstacle_collisions;

} USERDATA;

extern USERDATA *mydata;

