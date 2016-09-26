#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialConvolution.c"
#else

void THNN_(SpatialConvolution_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          int dW,
          int dH)
{
  int dimw = 2;
  int dimh = 1;

  THArgCheck(input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D(batch mode) tensor expected");

  if (input->nDimension == 4) {
    dimw++;
    dimh++;
  }

  {
    long nOutputPlane = weight->size[0];
    long kW           = weight->size[3];
    long kH           = weight->size[2];
    long inputWidth   = input->size[dimw];
    long inputHeight  = input->size[dimh];
    long outputWidth  = (inputWidth - kW) / dW + 1;
    long outputHeight = (inputHeight - kH) / dH + 1;

    if (input->nDimension == 3)
    {
      long i;
      real* bias_data;
      real* output_data;

      THTensor_(resize3d)(output, nOutputPlane, outputHeight, outputWidth);
      /* add bias */
      bias_data = THTensor_(data)(bias);
      output_data = THTensor_(data)(output);

#pragma omp parallel for private(i)
      for (i=0; i<bias->size[0]; i++)
      {
        /*THTensor_(select)(outn,output,0,i);*/
        /*TH_TENSOR_APPLY(real,outn, *outn_data = bias_data[i];);*/
        real *ptr_output = output_data + i*outputWidth*outputHeight;
        long j;
        for(j = 0; j < outputWidth*outputHeight; j++)
          ptr_output[j] = bias_data[i];
      }
      /*THTensor_(free)(outn);*/
      
      /* do convolutions */
      THTensor_(conv2Dmv)(output, 1.0, 1.0, input, weight, dH, dW, "V","X");
    }
    else
    {
      real* bias_data;
      real* output_data; 
      long p;

      THTensor_(resize4d)(output, input->size[0], nOutputPlane, outputHeight, outputWidth);
      
      bias_data = THTensor_(data)(bias);
      output_data = THTensor_(data)(output);
      
#pragma omp parallel for private(p)
      for (p=0; p<input->size[0]; p++)
      {
        /* BIAS */
        long i;
        for (i=0; i<bias->size[0]; i++)
        {
          real *ptr_output = output_data + p*nOutputPlane*outputWidth*outputHeight + i*outputWidth*outputHeight;
          long j;
          for(j = 0; j < outputWidth*outputHeight; j++)
            ptr_output[j] = bias_data[i];
        }
      }
      
      /* do convolutions */
      THTensor_(conv2Dmm)(output, 1.0, 1.0, input, weight, dH, dW, "V","X");
    }
  }
}

#endif
