# Auditory Attention Decoding

This branch is the software release for the 2019 paper: https://www.nature.com/articles/s41598-019-47795-0

====== LICENSE.txt =========

Patented Algorithm with Software Research Purposes License Agreement


This Agreement, effective as of the date the software is downloaded (“EFFECTIVE DATE”) is between the Massachusetts Institute of Technology ("MIT"), a non-profit institution of higher education, and you (“YOU”). 

WHEREAS, MIT has authored certain copyrighted software pertaining to MIT Case No. 21955L, "End to End Deep Neural Network for Auditory Attention Decoding", by Paul T. Calamia, Gregory Alan. Ciccarelli, Stephanie Haro, Michael Arthur. Nolan, Thomas F. Quatieri and Christopher Smalt (“PROGRAM”); and

WHEREAS, MIT also owns and holds certain rights, title, and interest to a filed patent application for the technology described in MIT Case No. 21085L, "End to End Deep Neural Network for Auditory Attention Decoding", by Michael S. Brandstein, Paul T. Calamia, Gregory Alan. Ciccarelli, Stephanie Haro, Michael Arthur. Nolan, Thomas F. Quatieri and Christopher Smalt, United States of America Serial No. 16/720810, Filed December 19, 2019 (“PATENT RIGHTS"); and

WHEREAS, MIT desires to distribute the PROGRAM for academic, non-commercial research use and testing as well as to raise awareness of the PATENT RIGHTS to promote commercial adoption, it hereby agrees to grant a limited license to the PROGRAM for research and non-commercial purposes only, with MIT retaining all other rights in the PATENT RIGHTS and in the PROGRAM; and

THEREFORE, MIT agrees to make the PROGRAM available to YOU, subject to the following terms and conditions: 


1.  Grant.   

(a) Subject to the terms of this Agreement, MIT hereby grants YOU a royalty-free, non-transferable, non-exclusive worldwide license under its copyright to use, reproduce, modify, publicly display and perform the PROGRAM solely for non-commercial research and/or academic testing purposes.  

(b) In order to obtain any further license rights, including the right to use the PROGRAM or PATENT RIGHTS for commercial purposes, (including industrially sponsored research), YOU must contact MIT’s Technology Licensing Office about additional commercial license agreements. 

2.  Disclaimer.   THE PROGRAM MADE AVAILABLE HEREUNDER IS  "AS IS", WITHOUT WARRANTY OF ANY KIND EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, NOR REPRESENTATION THAT THE PROGRAM DOES NOT INFRINGE THE  INTELLECTUAL PROPERTY RIGHTS OF ANY THIRD PARTY. MIT has no obligation to assist in your installation or use of the PROGRAM or to provide services or maintenance of any type with respect to the PROGRAM.  The entire risk as to the quality and performance of the PROGRAM is borne by YOU.  YOU acknowledge that the PROGRAM may contain errors or bugs.  YOU must determine whether the PROGRAM sufficiently meets your requirements.  This disclaimer of warranty constitutes an essential part of this Agreement.

3. No Consequential Damages; Indemnification.  IN NO EVENT SHALL MIT BE LIABLE TO YOU FOR ANY LOST PROFITS OR OTHER INDIRECT, PUNITIVE, INCIDENTAL OR CONSEQUENTIAL DAMAGES RELATING TO THE SUBJECT MATTER OF THIS AGREEMENT.  

4. Copyright.  YOU agree to retain MIT's copyright notice on all copies of the PROGRAM or portions thereof.

5.  Term.    The Term of this Agreement shall be ten years from the date YOU accept the terms of this license.  

6. Export Control.   YOU agree to comply with all United States export control laws and regulations controlling the export of the PROGRAM, including, without limitation, all Export Administration Regulations of the United States Department of Commerce.  Among other things, these laws and regulations prohibit, or require a license for, the export of certain types of software to specified countries.

7. Notices or Additional Licenses.  Any notice, communication or commercial license requests shall be directed to:

        Massachusetts Institute of Technology
        Technology Licensing Office, Rm NE18-501
        255 Main Street 
        Cambridge, MA  02142
        (617) 253 – 6966

ATTENTION:  	Daniel Dardani, Technology Licensing Officer
                ddardani@mit.edu

						

8.  General.  This Agreement shall be governed by the laws of the Commonwealth of Massachusetts.  The parties acknowledge that this Agreement sets forth the entire Agreement and understanding of the parties as to the subject matter.

CLICKING ON THE "ACCEPT" BUTTON OR PROCEEDING WITH THE SOFTWARE DOWNLOAD CONSTITUTES ACCEPTANCE OF THIS AGREEMENT.  YOU ARE CONSENTING TO BE BOUND BY ALL OF THE TERMS OF THIS AGREEMENT.  IF YOU DO NOT AGREE TO ALL THE TERMS OF THIS AGREEMENT, CLICK THE "DO NOT ACCEPT" BUTTON, OR DISCONTINUE THE INSTALLATION/DOWNLOAD PROCESS.  

===============

Copyright 2019 Massachusetts Institute of Technology

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.

© 2019 Massachusetts Institute of Technology.

MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.



#### ----------------------------------- Revision 0001 ------------------------------------

## Code release:

### Least Squares

 /LLlsq_grid_sklearn.ipynb   
 /lsq_grid.py


### DNN   
 /nn_eeg2env.ipynb

#### DNN- de Taillez
 /201806190841_1hid_b_tanh_b_htanh.py

Wet
 /201806221952_get_noscale.py   
 /201905031437_get_data_recon_wet2dry.py
 
Dry
 /201807141456_get_noscale_dsi.py 

#### DNN- binary clf

Wet   
 /201809262034_binary_conv.py  
 /201809262022_get_binary_conv.py   
 /201905031434_get_data_bce_wet2dry.py  

Dry   
 /201809272028_binary_conv_dsi.py  
 /201809272008_get_binary_conv_dry.py   

 






