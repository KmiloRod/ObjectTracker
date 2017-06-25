   classdef HOGNSSFeatures
   %HOGNSSFeatures Wrapper class for C++ class Image
   %   obj = HOGNSSFeatures(imSize, varargin)
   %   creates an object with properties 
   %
   %   Properties:
   %   winSize          - Size of window in pixels
   %   cellSize         - Size of cells in pixels
   %   blockSize        - Size of block in cells
   %   blockOverlap     - Overlap of blocks in cells
   %
   %   features = compute(obj, img) computes HOG features on
   %   input image, img, for the object defined by obj.
   %
   %   release(obj) releases object memory.

       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       %  Properties
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
       properties
           winSize = cast([1 1],'int32');
           cellSize = cast([8 8],'int32');
           blockSize = cast([2 2],'int32');
           blockOverlap = cast([1 1],'int32');        
       end
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       %  Public methods
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
       methods
           % Constructor
           function obj = HOGNSSFeatures(imSize, varargin)
               if mod(nargin-1,2) == 1
                   error('Incorrect number of inputs');
               end
               p = getInputParser(obj);
               parse(p,varargin{:});
               userInput = p.Results;
               params = HOGNSSFeatures.setparams(userInput,imSize);
               HOGNSSFeaturesOCV('construct', params);
           end

           % Get HOG features vector
           function features = compute(~,img, NSS, Cs)
               img = uint8(img);
               NSS = logical(NSS);
               Cs = single(Cs);
               % Get HOG features vector
               features = HOGNSSFeaturesOCV('compute', img, NSS, Cs);
           end          
           
           % Release object memory
           function release(~)
               % free persistent memory for model
               HOGNSSFeaturesOCV('destroy');
           end
       end
       methods (Access = private)
           % Input parser
           function parser = getInputParser(obj)
               persistent p
               if isempty(p)
                   p = inputParser();
                   addParameter(p, 'CellSize',     obj.cellSize);
                   addParameter(p, 'BlockSize',    obj.blockSize);
                   addParameter(p, 'BlockOverlap', obj.blockOverlap);
               end
               parser = p;
           end
       end
       methods (Static)
           % Set parameters
           function params = setparams(userInput, sz)
               cell_Size = userInput.CellSize;
               win_Size = floor([sz(1)./cell_Size(1) sz(2)./cell_Size(2)]);
               win_Size = win_Size.*cell_Size;
               params.CellSize     = int32(userInput.CellSize);
               params.BlockSize    = int32(userInput.BlockSize);
               params.BlockOverlap = int32(userInput.BlockOverlap);
               params.ImageSize  = int32(sz(1:2));
               params.WindowSize = int32(win_Size);
           end
       end
   end