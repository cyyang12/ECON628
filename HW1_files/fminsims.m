function [b,val,exit_code] = fminsims(func, b0, optimization_options)
  % This function is a wrapper function for the 'csminwel' optimization routine 
  % written by Chris Sims. The most recent version of Chris Sims' optimization codes 
  % can be found at http://sims.princeton.edu/yftp/optimize/. 
  %
  % This function minimizes function 'func' starting from the value
  % 'b0'. 'optimization_options' are used to specify certain parameters of 
  % the optimization procedure such as 'MaxIter' and 'TolFun'.
  % 
  % This function can use user supplied gradient. The gradient should be caluclated by
  % the function 'func' when two output arguments are requested, i.e., in the 
  % same way as it is done by the MATLAB function 'fminunc'. See the help for
  % the function 'fminunc'. To indicate that the function 'func' is able to provide 
  % analytic derivative one should set the option 'GradObj' to 'on' in the 'optimization_options'.
  % 
  % If the optimization ends abnormally, it is possible to retrieve the current value, 
  % gradient, and Hessian from the files g1.mat and H.mat. See the details below.
  %
  % The wrapper functions 'fminsims', 'fminsims_crit', and 'fminsims_crit_grad' are
  % written by Kirill Evdokimov, but all the optimization functions are
  % written by Chris Sims and are taken directly from his website.
  %
  % This version: 03/22/2011
  %
  %
  % Copyright (C) 2010  Christopher Sims
  %
  % This program can be redistributed and/or modified under the terms of the 
  % GNU General Public License, version 3 or later.
  %
  % For details of the license, see <http://www.gnu.org/licenses/>.
  % 
  
  h = eye(length(b0)); %initial hessian
  
  if (nargin<2 || nargin>3)
    error('fminsims: Wrong number of parameters.');
  end
  
  global fminsims_func %this is the user supplied function to minimize
  fminsims_func = func;
  
  %extract optimization parameters from 'optimization_options'
  use_quick_grad_string = optimget(optimization_options, 'GradObj', 'off');
  if strcmpi(use_quick_grad_string,'on')
    grad_fn_name = 'sims_crit_grad'; %using analytic "quick gradient"
  else
    grad_fn_name = []; %not using the "quick gradient" method
  end 
  max_iter = optimget(optimization_options, 'MaxIter', 10000);
  tol_fun = optimget(optimization_options, 'TolFun', sqrt(eps(1.0)));
  
  global fminsims_exit_code
  [val,b,ignore1,ignore2,ignore3,ignore4,fminsims_exit_code] = csminwel('sims_crit',b0,h, grad_fn_name,tol_fun, max_iter);  %#ok<ASGLU>
  
  if fminsims_exit_code==0
    exit_code = 1; %no error
  else
    exit_code = -111; %some error. You can check the value of the 
          %  global variable 'fminsims_exit_code' for details. Its values 
          %  have the following interpretations:
          %  1  : zero gradient
          % 2,4 : back and forth on step length never finished
          %  3  : smallest step still improving too slow
          %  5  : largest step still improving too fast
          %  6  : smallest step still improving too slow, reversed gradient
          %  7  : warning: possible inaccuracy in H matrix
  end
end

function val = sims_crit(b)
  % Criterion function. b must be p x k, where p is the number of
  % parameters, and k is the number of different parameter vectors to
  % consider. This function needs to be able to take a matrix of parameters and
  % calculate a row-vector of values to be compatible with the numeric
  % derivative procedure used by csminwel.
  k = size(b,2); %how many points need to be evaluated
  val = zeros(1,k);
  
  global fminsims_func %this is the user supplied function to minimize
  for i=1:k
    val(i) = fminsims_func(b(:,k));
  end
end


function [g, badg] = sims_crit_grad(b)
  global fminsims_func %this is the user supplied function to minimize
  [v,g] = fminsims_func(b);
  badg = ~(isfinite(v) && all(isfinite(g(:))) && abs(v)-sqrt(realmax)<=eps(1.0) && all(abs(g(:))-sqrt(realmax)<=eps(1.0)) ); 
    %too big values of v or g are taken as an indication of the derivative being "bad"
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The functions below are written by Chris Sims and are taken from 
% his website: http://sims.princeton.edu/yftp/optimize/
%
% The functions 'csminwel', 'csminit', 'numgrad', 'bfgsi', and 'csolve' 
% are downloaded from the above website and combined in this file.
%
% KE: I have wrapped the 'save' statements into 'try/catch/end' sequence,
%       in order to avoid program interuption due to some IO issues, such as
%       not having the permissions to write a file on disk.

function [fh,xh,gh,H,itct,fcount,retcodeh] = csminwel(fcn,x0,H0,grad,crit,nit,varargin)
%[fhat,xhat,ghat,Hhat,itct,fcount,retcodehat] = csminwel(fcn,x0,H0,grad,crit,nit,varargin)
% fcn:   string naming the objective function to be minimized
% x0:    initial value of the parameter vector
% H0:    initial value for the inverse Hessian.  Must be positive definite.
% grad:  Either a string naming a function that calculates the gradient, or the null matrix.
%        If it's null, the program calculates a numerical gradient.  In this case fcn must
%        be written so that it can take a matrix argument and produce a row vector of values.
% crit:  Convergence criterion.  Iteration will cease when it proves impossible to improve the
%        function value by more than crit.
% nit:   Maximum number of iterations.
% varargin: A list of optional length of additional parameters that get handed off to fcn each
%        time it is called.
%        Note that if the program ends abnormally, it is possible to retrieve the current x,
%        f, and H from the files g1.mat and H.mat that are written at each iteration and at each
%        hessian update, respectively.  (When the routine hits certain kinds of difficulty, it
%        write g2.mat and g3.mat as well.  If all were written at about the same time, any of them
%        may be a decent starting point.  One can also start from the one with best function value.)
[nx,no]=size(x0);
nx=max(nx,no);
Verbose=1;
NumGrad= isempty(grad);
done=0;
itct=0;
fcount=0;
snit=100;
%tailstr = ')';
%stailstr = [];
% Lines below make the number of Pi's optional.  This is inefficient, though, and precludes
% use of the matlab compiler.  Without them, we use feval and the number of Pi's must be
% changed with the editor for each application.  Places where this is required are marked
% with ARGLIST comments
%for i=nargin-6:-1:1
%   tailstr=[ ',P' num2str(i)  tailstr];
%   stailstr=[' P' num2str(i) stailstr];
%end
f0 = feval(fcn,x0,varargin{:});
%ARGLIST
%f0 = feval(fcn,x0,P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12,P13);
% disp('first fcn in csminwel.m ----------------') % Jinill on 9/5/95
if f0 > 1e50, disp('Bad initial parameter.'), return, end
if NumGrad
   if length(grad)==0
      [g badg] = numgrad(fcn,x0, varargin{:});
      %ARGLIST
      %[g badg] = numgrad(fcn,x0,P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12,P13);
   else
      badg=any(find(grad==0));
      g=grad;
   end
   %numgrad(fcn,x0,P1,P2,P3,P4);
else
   [g badg] = feval(grad,x0,varargin{:});
   %ARGLIST
   %[g badg] = feval(grad,x0,P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12,P13);
end
retcode3=101;
x=x0;
f=f0;
H=H0;
cliff=0;
while ~done
   g1=[]; g2=[]; g3=[];
   %addition fj. 7/6/94 for control
   disp('-----------------')
   disp('-----------------')
   %disp('f and x at the beginning of new iteration')
   disp(sprintf('f at the beginning of new iteration, %20.10f',f))
   %-----------Comment out this line if the x vector is long----------------
      disp([sprintf('x = ') sprintf('%15.8g %15.8g %15.8g %15.8g\n',x)]);
   %-------------------------
   itct=itct+1;
   [f1 x1 fc retcode1] = csminit(fcn,x,f,g,badg,H,varargin{:});
   %ARGLIST
   %[f1 x1 fc retcode1] = csminit(fcn,x,f,g,badg,H,P1,P2,P3,P4,P5,P6,P7,...
   %           P8,P9,P10,P11,P12,P13);
   % itct=itct+1;
   fcount = fcount+fc;
   % erased on 8/4/94
   % if (retcode == 1) | (abs(f1-f) < crit)
   %    done=1;
   % end
   % if itct > nit
   %    done = 1;
   %    retcode = -retcode;
   % end
   if retcode1 ~= 1
      if retcode1==2 | retcode1==4
         wall1=1; badg1=1;
      else
         if NumGrad
            [g1 badg1] = numgrad(fcn, x1,varargin{:});
            %ARGLIST
            %[g1 badg1] = numgrad(fcn, x1,P1,P2,P3,P4,P5,P6,P7,P8,P9,...
            %                P10,P11,P12,P13);
         else
            [g1 badg1] = feval(grad,x1,varargin{:});
            %ARGLIST
            %[g1 badg1] = feval(grad, x1,P1,P2,P3,P4,P5,P6,P7,P8,P9,...
            %                P10,P11,P12,P13);
         end
         wall1=badg1;
         % g1
         try %Try/catch added by KE
           save g1 g1 x1 f1 varargin;
         catch %do nothing
         end
           %ARGLIST
         %save g1 g1 x1 f1 P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 P12 P13;
      end
      if wall1 % & (~done) by Jinill
         % Bad gradient or back and forth on step length.  Possibly at
         % cliff edge.  Try perturbing search direction.
         %
         %fcliff=fh;xcliff=xh;
         Hcliff=H+diag(diag(H).*rand(nx,1));
         disp('Cliff.  Perturbing search direction.')
         [f2 x2 fc retcode2] = csminit(fcn,x,f,g,badg,Hcliff,varargin{:});
         %ARGLIST
         %[f2 x2 fc retcode2] = csminit(fcn,x,f,g,badg,Hcliff,P1,P2,P3,P4,...
         %     P5,P6,P7,P8,P9,P10,P11,P12,P13);
         fcount = fcount+fc; % put by Jinill
         if  f2 < f
            if retcode2==2 | retcode2==4
                  wall2=1; badg2=1;
            else
               if NumGrad
                  [g2 badg2] = numgrad(fcn, x2,varargin{:});
                  %ARGLIST
                  %[g2 badg2] = numgrad(fcn, x2,P1,P2,P3,P4,P5,P6,P7,P8,...
                  %      P9,P10,P11,P12,P13);
               else
                  [g2 badg2] = feval(grad,x2,varargin{:});
                  %ARGLIST
                  %[g2 badg2] = feval(grad,x2,P1,P2,P3,P4,P5,P6,P7,P8,...
                  %      P9,P10,P11,P12,P13);
               end
               wall2=badg2;
               % g2
               badg2
               try %Try/catch added by KE
                 save g2 g2 x2 f2 varargin
               catch %do nothing
               end
               %ARGLIST
               %save g2 g2 x2 f2 P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 P12 P13;
            end
            if wall2
               disp('Cliff again.  Try traversing')
               if norm(x2-x1) < 1e-13
                  f3=f; x3=x; badg3=1;retcode3=101;
               else
                  gcliff=((f2-f1)/((norm(x2-x1))^2))*(x2-x1);
                  if(size(x0,2)>1), gcliff=gcliff', end
                  [f3 x3 fc retcode3] = csminit(fcn,x,f,gcliff,0,eye(nx),varargin{:});
                  %ARGLIST
                  %[f3 x3 fc retcode3] = csminit(fcn,x,f,gcliff,0,eye(nx),P1,P2,P3,...
                  %         P4,P5,P6,P7,P8,...
                  %      P9,P10,P11,P12,P13);
                  fcount = fcount+fc; % put by Jinill
                  if retcode3==2 | retcode3==4
                     wall3=1; badg3=1;
                  else
                     if NumGrad
                        [g3 badg3] = numgrad(fcn, x3,varargin{:});
                        %ARGLIST
                        %[g3 badg3] = numgrad(fcn, x3,P1,P2,P3,P4,P5,P6,P7,P8,...
                        %                        P9,P10,P11,P12,P13);
                     else
                        [g3 badg3] = feval(grad,x3,varargin{:});
                        %ARGLIST
                        %[g3 badg3] = feval(grad,x3,P1,P2,P3,P4,P5,P6,P7,P8,...
                        %                         P9,P10,P11,P12,P13);
                     end
                     wall3=badg3;
                     % g3
                     badg3
                     try %Try/catch added by KE
                       save g3 g3 x3 f3 varargin;
                     catch %do nothing
                     end
                     %ARGLIST
                     %save g3 g3 x3 f3 P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 P12 P13;
                  end
               end
            else
               f3=f; x3=x; badg3=1; retcode3=101;
            end
         else
            f3=f; x3=x; badg3=1;retcode3=101;
         end
      else
         % normal iteration, no walls, or else we're finished here.
         f2=f; f3=f; badg2=1; badg3=1; retcode2=101; retcode3=101;
      end
   else 
      f2=f;f3=f;f1=f;retcode2=retcode1;retcode3=retcode1;
   end
   %how to pick gh and xh
   if f3 < f - crit & badg3==0
      ih=3
      fh=f3;xh=x3;gh=g3;badgh=badg3;retcodeh=retcode3;
   elseif f2 < f - crit & badg2==0
      ih=2
      fh=f2;xh=x2;gh=g2;badgh=badg2;retcodeh=retcode2;
   elseif f1 < f - crit & badg1==0
      ih=1
      fh=f1;xh=x1;gh=g1;badgh=badg1;retcodeh=retcode1;
   else
      [fh,ih] = min([f1,f2,f3]);
      disp(sprintf('ih = %d',ih))
      %eval(['xh=x' num2str(ih) ';'])
      switch ih
         case 1
            xh=x1;
         case 2
            xh=x2;
         case 3
            xh=x3;
      end %case
      %eval(['gh=g' num2str(ih) ';'])
      %eval(['retcodeh=retcode' num2str(ih) ';'])
      retcodei=[retcode1,retcode2,retcode3];
      retcodeh=retcodei(ih);
      if exist('gh')
         nogh=isempty(gh);
      else
         nogh=1;
      end
      if nogh
         if NumGrad
            [gh badgh] = numgrad(fcn,xh,varargin{:});
         else
            [gh badgh] = feval(grad, xh,varargin{:});
         end
      end
      badgh=1;
   end
   %end of picking
   %ih
   %fh
   %xh
   %gh
   %badgh
   stuck = (abs(fh-f) < crit);
   if (~badg)&(~badgh)&(~stuck)
      H = bfgsi(H,gh-g,xh-x);
   end
   if Verbose
      disp('----')
      disp(sprintf('Improvement on iteration %d = %18.9f',itct,f-fh))
   end
   % if Verbose
      if itct > nit
         disp('iteration count termination')
         done = 1;
      elseif stuck
         disp('improvement < crit termination')
         done = 1;
      end
      rc=retcodeh;
      if rc == 1
         disp('zero gradient')
      elseif rc == 6
         disp('smallest step still improving too slow, reversed gradient')
      elseif rc == 5
         disp('largest step still improving too fast')
      elseif (rc == 4) | (rc==2)
         disp('back and forth on step length never finished')
      elseif rc == 3
         disp('smallest step still improving too slow')
      elseif rc == 7
         disp('warning: possible inaccuracy in H matrix')
      end
   % end
   f=fh;
   x=xh;
   g=gh;
   badg=badgh;
end
% what about making an m-file of 10 lines including numgrad.m
% since it appears three times in csminwel.m
end

%%% Function csminit - it used to be in a separate file csminit.m %%%
function [fhat,xhat,fcount,retcode] = csminit(fcn,x0,f0,g0,badg,H0,varargin)
% [fhat,xhat,fcount,retcode] = csminit(fcn,x0,f0,g0,badg,H0,...
%                                       P1,P2,P3,P4,P5,P6,P7,P8)
% retcodes: 0, normal step.  5, largest step still improves too fast.
% 4,2 back and forth adjustment of stepsize didn't finish.  3, smallest
% stepsize still improves too slow.  6, no improvement found.  1, zero
% gradient.
%---------------------
% Modified 7/22/96 to omit variable-length P list, for efficiency and compilation.
% Places where the number of P's need to be altered or the code could be returned to
% its old form are marked with ARGLIST comments.
%
% Fixed 7/17/93 to use inverse-hessian instead of hessian itself in bfgs
% update.
%
% Fixed 7/19/93 to flip eigenvalues of H to get better performance when
% it's not psd.
%
%tailstr = ')';
%for i=nargin-6:-1:1
%   tailstr=[ ',P' num2str(i)  tailstr];
%end
%ANGLE = .03;
ANGLE = .005;
%THETA = .03;
THETA = .3; %(0<THETA<.5) THETA near .5 makes long line searches, possibly fewer iterations.
FCHANGE = 1000;
MINLAMB = 1e-9;
% fixed 7/15/94
% MINDX = .0001;
% MINDX = 1e-6;
MINDFAC = .01;
fcount=0;
lambda=1;
xhat=x0;
f=f0;
fhat=f0;
g = g0;
gnorm = norm(g);
%
if (gnorm < 1.e-12) & ~badg % put ~badg 8/4/94
   retcode =1;
   dxnorm=0;
   % gradient convergence
else
   % with badg true, we don't try to match rate of improvement to directional
   % derivative.  We're satisfied just to get some improvement in f.
   %
   %if(badg)
   %   dx = -g*FCHANGE/(gnorm*gnorm);
   %  dxnorm = norm(dx);
   %  if dxnorm > 1e12
   %     disp('Bad, small gradient problem.')
   %     dx = dx*FCHANGE/dxnorm;
   %   end
   %else
   % Gauss-Newton step;
   %---------- Start of 7/19/93 mod ---------------
   %[v d] = eig(H0);
   %toc
   %d=max(1e-10,abs(diag(d)));
   %d=abs(diag(d));
   %dx = -(v.*(ones(size(v,1),1)*d'))*(v'*g);
%      toc
   dx = -H0*g;
%      toc
   dxnorm = norm(dx);
   if dxnorm > 1e12
      disp('Near-singular H problem.')
      dx = dx*FCHANGE/dxnorm;
   end
   dfhat = dx'*g0;
   %end
   %
   %
   if ~badg
      % test for alignment of dx with gradient and fix if necessary
      a = -dfhat/(gnorm*dxnorm);
      if a<ANGLE
         dx = dx - (ANGLE*dxnorm/gnorm+dfhat/(gnorm*gnorm))*g;
         % suggested alternate code:  ---------------------
         dx = dx*dxnorm/norm(dx)    % This keeps scale invariant to the angle correction
         % ------------------------------------------------
         dfhat = dx'*g;
         % dxnorm = norm(dx);  % this line unnecessary with modification that keeps scale invariant
         disp(sprintf('Correct for low angle: %g',a))
      end
   end
   disp(sprintf('Predicted improvement: %18.9f',-dfhat/2))
   %
   % Have OK dx, now adjust length of step (lambda) until min and
   % max improvement rate criteria are met.
   done=0;
   factor=3;
   shrink=1;
   lambdaMin=0;
   lambdaMax=inf;
   lambdaPeak=0;
   fPeak=f0;
   lambdahat=0;
   while ~done
      if size(x0,2)>1
         dxtest=x0+dx'*lambda;
      else
         dxtest=x0+dx*lambda;
      end
      % home
      f = feval(fcn,dxtest,varargin{:});
      %ARGLIST
      %f = feval(fcn,dxtest,P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12,P13);
      % f = feval(fcn,x0+dx*lambda,P1,P2,P3,P4,P5,P6,P7,P8);
      disp(sprintf('lambda = %10.5g; f = %20.7f',lambda,f ))
      %debug
      %disp(sprintf('Improvement too great? f0-f: %g, criterion: %g',f0-f,-(1-THETA)*dfhat*lambda))
      if f<fhat
         fhat=f;
         xhat=dxtest;
         lambdahat = lambda;
      end
      fcount=fcount+1;
      shrinkSignal = (~badg & (f0-f < max([-THETA*dfhat*lambda 0]))) | (badg & (f0-f) < 0) ;
      growSignal = ~badg & ( (lambda > 0)  &  (f0-f > -(1-THETA)*dfhat*lambda) );
      if  shrinkSignal  &   ( (lambda>lambdaPeak) | (lambda<0) )
         if (lambda>0) & ((~shrink) | (lambda/factor <= lambdaPeak))
            shrink=1;
            factor=factor^.6;
            while lambda/factor <= lambdaPeak
               factor=factor^.6;
            end
            %if (abs(lambda)*(factor-1)*dxnorm < MINDX) | (abs(lambda)*(factor-1) < MINLAMB)
            if abs(factor-1)<MINDFAC
               if abs(lambda)<4
                  retcode=2;
               else
                  retcode=7;
               end
               done=1;
            end
         end
         if (lambda<lambdaMax) & (lambda>lambdaPeak)
            lambdaMax=lambda;
         end
         lambda=lambda/factor;
         if abs(lambda) < MINLAMB
            if (lambda > 0) & (f0 <= fhat)
               % try going against gradient, which may be inaccurate
               lambda = -lambda*factor^6
            else
               if lambda < 0
                  retcode = 6;
               else
                  retcode = 3;
               end
               done = 1;
            end
         end
      elseif  (growSignal & lambda>0) |  (shrinkSignal & ((lambda <= lambdaPeak) & (lambda>0)))
         if shrink
            shrink=0;
            factor = factor^.6;
            %if ( abs(lambda)*(factor-1)*dxnorm< MINDX ) | ( abs(lambda)*(factor-1)< MINLAMB)
            if abs(factor-1)<MINDFAC
               if abs(lambda)<4
                  retcode=4;
               else
                  retcode=7;
               end
               done=1;
            end
         end
         if ( f<fPeak ) & (lambda>0)
            fPeak=f;
            lambdaPeak=lambda;
            if lambdaMax<=lambdaPeak
               lambdaMax=lambdaPeak*factor*factor;
            end
         end
         lambda=lambda*factor;
         if abs(lambda) > 1e20;
            retcode = 5;
            done =1;
         end
      else
         done=1;
         if factor < 1.2
            retcode=7;
         else
            retcode=0;
         end
      end
   end
end
disp(sprintf('Norm of dx %10.5g', dxnorm))
end


%%% Function csolve - it used to be in a separate file csolve.m %%%
function [x,rc] = csolve(FUN,x,gradfun,crit,itmax,varargin)
%function [x,rc] = csolve(FUN,x,gradfun,crit,itmax,varargin)
% FUN should be written so that any parametric arguments are packed in to x,
% and so that if presented with a matrix x, it produces a return value of
% same dimension of x.  The number of rows in x and FUN(x) are always the
% same.  The number of columns is the number of different input arguments
% at which FUN is to be evaluated.
%
% gradfun:  string naming the function called to evaluate the gradient matrix.  If this
%           is null (i.e. just "[]"), a numerical gradient is used instead.
% crit:     if the sum of absolute values that FUN returns is less than this,
%           the equation is solved.
% itmax:    the solver stops when this number of iterations is reached, with rc=4
% varargin: in this position the user can place any number of additional arguments, all
%           of which are passed on to FUN and gradfun (when it is non-empty) as a list of 
%           arguments following x.
% rc:       0 means normal solution, 1 and 3 mean no solution despite extremely fine adjustments
%           in step length (very likely a numerical problem, or a discontinuity). 4 means itmax
%           termination.
%---------- delta --------------------
% differencing interval for numerical gradient
delta = 1e-6;
%-------------------------------------
%------------ alpha ------------------
% tolerance on rate of descent
alpha=1e-3;
%---------------------------------------
%------------ verbose ----------------
verbose=1;% if this is set to zero, all screen output is suppressed
%-------------------------------------
%------------ analyticg --------------
analyticg=1-isempty(gradfun); %if the grad argument is [], numerical derivatives are used.
%-------------------------------------
nv=length(x);
tvec=delta*eye(nv);
done=0;
if isempty(varargin)
   f0=feval(FUN,x);
else
   f0=feval(FUN,x,varargin{:});
end   
af0=sum(abs(f0));
af00=af0;
itct=0;
while ~done
   if itct>3 & af00-af0<crit*max(1,af0) & rem(itct,2)==1
      randomize=1;
   else
      if ~analyticg
         if isempty(varargin)
            grad = (feval(FUN,x*ones(1,nv)+tvec)-f0*ones(1,nv))/delta;
         else
            grad = (feval(FUN,x*ones(1,nv)+tvec,varargin{:})-f0*ones(1,nv))/delta;
         end
      else % use analytic gradient
         grad=feval(gradfun,x,varargin{:});
      end
      if isreal(grad)
         if rcond(grad)<1e-12
            grad=grad+tvec;
         end
         dx0=-grad\f0;
         randomize=0;
      else
         if(verbose),disp('gradient imaginary'),end
         randomize=1;
      end
   end
   if randomize
      if(verbose),fprintf(1,'\n Random Search'),end
      dx0=norm(x)./randn(size(x));
   end
   lambda=1;
   lambdamin=1;
   fmin=f0;
   xmin=x;
   afmin=af0;
   dxSize=norm(dx0);
   factor=.6;
   shrink=1;
   subDone=0;
   while ~subDone
      dx=lambda*dx0;
      f=feval(FUN,x+dx,varargin{:});
      af=sum(abs(f));
      if af<afmin
         afmin=af;
         fmin=f;
         lambdamin=lambda;
         xmin=x+dx;
      end
      if ((lambda >0) & (af0-af < alpha*lambda*af0)) | ((lambda<0) & (af0-af < 0) )
         if ~shrink
            factor=factor^.6;
            shrink=1;
         end
         if abs(lambda*(1-factor))*dxSize > .1*delta;
            lambda = factor*lambda;
         elseif (lambda > 0) & (factor==.6) %i.e., we've only been shrinking
            lambda=-.3;
         else %
            subDone=1;
            if lambda > 0
               if factor==.6
                  rc = 2;
               else
                  rc = 1;
               end
            else
               rc=3;
            end
         end
      elseif (lambda >0) & (af-af0 > (1-alpha)*lambda*af0)
         if shrink
            factor=factor^.6;
            shrink=0;
         end
         lambda=lambda/factor;
      else % good value found
         subDone=1;
         rc=0;
      end
   end % while ~subDone
   itct=itct+1;
   if(verbose)
      fprintf(1,'\nitct %d, af %g, lambda %g, rc %g',itct,afmin,lambdamin,rc)
      fprintf(1,'\n   x  %10g %10g %10g %10g',xmin);
      fprintf(1,'\n   f  %10g %10g %10g %10g',fmin);
   end
   x=xmin;
   f0=fmin;
   af00=af0;
   af0=afmin;
   if itct >= itmax
      done=1;
      rc=4;
   elseif af0<crit;
      done=1;
      rc=0;
   end
end
end


%%% Function bfgsi - it used to be in a separate file bfgsi.m %%%
function H = bfgsi(H0,dg,dx)
% H = bfgsi(H0,dg,dx)
% dg is previous change in gradient; dx is previous change in x;
% 6/8/93 version that updates inverse hessian instead of hessian
% itself.
% Copyright by Christopher Sims 1996.  This material may be freely
% reproduced and modified.
if size(dg,2)>1
   dg=dg';
end
if size(dx,2)>1
   dx=dx';
end
Hdg = H0*dg;
dgdx = dg'*dx;
if (abs(dgdx) >1e-12)
   H = H0 + (1+(dg'*Hdg)/dgdx)*(dx*dx')/dgdx - (dx*Hdg'+Hdg*dx')/dgdx;
else
   disp('bfgs update failed.')
   disp(['|dg| = ' num2str(sqrt(dg'*dg)) '|dx| = ' num2str(sqrt(dx'*dx))]);
   disp(['dg''*dx = ' num2str(dgdx)])
   disp(['|H*dg| = ' num2str(Hdg'*Hdg)])
   H=H0;
end

try %Try/catch added by KE
  save H.dat H
catch %do nothing
end
end

%%% Function numgrad - it used to be in a separate file numgrad.m %%%
function [g, badg] = numgrad(fcn,x,varargin)
% function [g badg] = numgrad(fcn,x,varargin)
%
delta = 1e-6;
%delta=1e-2;
n=length(x);
tvec=delta*eye(n);
g=zeros(n,1);
%--------------------old way to deal with variable # of P's--------------
%tailstr = ')';
%stailstr = [];
%for i=nargin-2:-1:1
%   tailstr=[ ',P' num2str(i)  tailstr];
%   stailstr=[' P' num2str(i) stailstr];
%end
%f0 = eval([fcn '(x' tailstr]); % Is there a way not to do this?
%---------------------------------------------------------------^yes
f0 = feval(fcn,x,varargin{:});
% disp(' first fcn in numgrad.m ------------------')
%home
% disp('numgrad.m is working. ----') % Jiinil on 9/5/95
% sizex=size(x),sizetvec=size(tvec),x,    % Jinill on 9/6/95
badg=0;
for i=1:n
   scale=1; % originally 1
   % i,tveci=tvec(:,i)% ,plus=x+scale*tvec(:,i) % Jinill Kim on 9/6/95
   if size(x,1)>size(x,2)
      tvecv=tvec(i,:);
   else
      tvecv=tvec(:,i);
   end
   g0 = (feval(fcn,x+scale*tvecv', varargin{:}) - f0) ...
         /(scale*delta);
   % disp(' fcn in the i=1:n loop of numgrad.m ------------------')% Jinill 9/6/95
   % disp('          and i is')               % Jinill
   % i                         % Jinill
   % fprintf('Gradient w.r.t. %3d: %10g\n',i,g0) %see below Jinill 9/6/95
% -------------------------- special code to essentially quit here
   % absg0=abs(g0) % Jinill on 9/6/95
   if abs(g0)< 1e15
      g(i)=g0;
      % disp('good gradient') % Jinill Kim
   else
      disp('bad gradient ------------------------') % Jinill Kim
      % fprintf('Gradient w.r.t. %3d: %10g\n',i,g0) %see above
      g(i)=0;
      badg=1;
      % return
      % can return here to save time if the gradient will never be
      % used when badg returns as true.
   end
end
%-------------------------------------------------------------
%     if g0 > 0
%        sided=2;
%        g1 = -(eval([fcn '(x-scale*tvec(:,i)''' tailstr]) - f0) ...
%           /(scale*delta);
%        if g1<0
%           scale = scale/10;
%        else
%           break
%        end
%     else
%        sided=1;
%        break
%     end
%  end
%  if sided==1
%     g(i)=g0;
%  else
%     if (g0<1e20)
%        if (g1>-1e20)
%           g(i)=(g0+g1)/2;
%        else
%           g(i)=0;
%           badg=1;
%           disp( ['Banging against wall, parameter ' int2str(i)] );
%        end
%     else
%        if g1>-1e20
%           if g1<0
%              g(i)=0;
%              badg=1;
%              disp( ['Banging against wall, parameter ' int2str(i)] );
%           else
%              g(i)=g1;
%           end
%        else
%           g(i)=0;
%           badg=1;
%           disp(['Valley around parameter ' int2str(i)])
%        end
%     end
%  end
%end
%save g.dat g x f0
%eval(['save g g x f0 ' stailstr]);
end

