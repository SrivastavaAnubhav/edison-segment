/////////////////////////////////////////////////////////////////////////////

// Name:        BgEdgeList.h

// Purpose:     BgEdgeList class

// Author:      Bogdan Georgescu

// Modified by:

// Created:     06/22/2000

// Copyright:   (c) Bogdan Georgescu

// Version:     v0.1

/////////////////////////////////////////////////////////////////////////////
#ifndef _BG_EDGE_LIST_H
#define _BG_EDGE_LIST_H

#include "BgEdge.h"

class BgEdgeList

{

public:

   int nEdges_;

   BgEdge* edgelist_;

   BgEdge* crtedge_;

   

   BgEdgeList();

   ~BgEdgeList();

   void AddEdge(float*, int);

   void AddEdge(int*, int nPoints);

   void RemoveShortEdges(int);

   void SetBinImage(BgImage*);

   bool SaveEdgeList(char*);

   void SetGradient(float*, float*, float*, int);

   void SetNoMark(void);

   void GetAllEdgePoints(int*, int*, int*);



};
#endif // !_BG_EDGE_LIST_H

