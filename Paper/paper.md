---
title: 'DL_Track: a python package to analyse muscle ultrasonography images'
tags:
  - Python
  - muscle
  - ultrasonography
  - muscle architecture
  - deep learning
authors:
  - name: Paul Ritsche
    orcid: 0000-0001-9446-7872
    corresponding: true
    affiliation: 1
  - name: Olivier Seynnes
    orcid: 0000-0002-1289-246X
    affiliation: 2
  - name: Neil Cronin
    orcid: 0000-0002-5332-1188
    affiliation: "3,4"
affiliations:
 - name: Department of Sport, Exercise and Health, University of Basel, Switzerland
   index: 1
 - name: Department for Physical Performance, Norwegian School of Sport Sciences, Oslo, Norway
   index: 2
 - name: University of Jyvaskyla, Faculty of Sport and Health Sciences, Jyvaskyla, Finland
   index: 3
 - name: School of Sport & Exercise, University of Gloucestershire, UK
   index: 4
date: 08 November 2022
bibliography: paper.bib
---

# Summary

Ultrasonography can be used to assess muscle architectural parameters during static and dynamic conditions. Nevertheless, the analysis of the acquired ultrasonography images presents a major difficulty. Muscle architectural parameters such as muscle thickness, fascicle length and pennation angle are mainly segmented manually. Manual analysis is time expensive, subjective and requires thorough expertise. Within recent years, several algorithms were developed to solve these issues. Yet, these are only partly automated, are not openly available, or lack in user friendliness. The DL_Track python package is designed to allow fully automated and rapid analyis of muscle architectural parameters in lower limb ultrasonography images.

# Statement of need

`DL_Track` is a python package to automatically segment and analyse muscle architectural parameters in longitudinal ultrasonography images and videos of human lower limb muscles. The use of ultrasonography to assess muscle morphological and architectural parameters is increasing in different scientific fields [@Naruse2022]. This is due to the high portability, cost-effectiveness and patient-friendlyness compared to other imaging modalities such as MRI. Muscle architectural parameters such as muscle thickness, fascicle length and pennation angle are used to assess muscular adaptations to training, disuse and ageing, especially in the lower limbs [@Sarto2021]. Albeit many disadvantages such as subjectivity, time expensiveness and required expertise, muscle architectural parameters in ultrasonography images are most often analyzed manually. A potential reason is the lack of versatile, accessible and easy to use analysis tools. Several algorithms to analyze muscle architectural parameters have been developed within the last years [@Cronin2011; @Rana2009; @Marzlinger2018; @Drazan2019; @Farris2016; @Seynnes2020]. Nonetheless, most algorithms only partly automate the analysis, introducing some subjectivity in the analysis process. Moreover, many analyse only single architectural parameters and can be exclusively used for image or video analysis. Most existing methods further rely on hardcoded image filtering processes developed on few example images. Given that image parameters do no suit the designed filters of the if the filters are improperly adjusted, these filtering processes likely fail. On top of that, most previously proposed algorithms lack in user friendliness as the provided code and usage documentation is limited.

`DL_Track` incorporates analysis modalities for fully automated analysis of ultrasonography images. `DL_Track` employs convolutional neural networks (CNNs) for the semantic segmentation of muscle fascicle fragments and aponeuroses. The employed CNNs consist of a VGG16 encoder path pre-trained on ImageNet [@Simonyan2015; @Deng2009] and a U-net decoder [@Ronneberger2015]. Based on the semantic segmentation, muscle thickness, fascicle length and pennation angle are calculated as the distance between the detected aponeuroses, the intersection between extrapolated fascicle trajectories and aponeuroses and the intersection angle between extrapolated fascicle trajectories and detected deep aponeuroses, respectively. The workflow of the `DL_Track` analysis algorithm is demonstrated in \autoref{fig:1}.

![Workflow of the DL_Track analysis algorithm.\label{fig:1}](figure1.png)

All `DL_Track` functionalities are embedded in a graphical user interface (GUI) to allow easy and intuitive usage. Apart from automated analysis, a manual analysis option included in the GUI as well. This is provided in case the provided pre-trained CNNs perform badly on input images or videos and subsequently avoid switching between softwares. Although we included images of the human gastrocnemius medialis, tibialis anterior, soleus and vastus lateralis from four different devices, it is likely the provided pre-trained CNNs fail when images from different muscles or devices are analyzed. However, an option to train CNNs bases on user data is also included in the GUI. Users are thereby enabled to train own CNNs based on own image or video data. The training ultrasonography image data, the pre-trained CNNs, example usage files as well as usage and testing instructions are provided.

# References
