%%
%% This is file `ppgccufmg.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% ppgccufmg.dtx  (with options: `class')
%% 
%% This is a generated file.
%% 
%% Copyright (C) 2004,2005,2006,2007,2008 by Vilar Camara Neto and Eduardo
%% Nakamura.
%% 
%% This file may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License, either
%% version 1.2 of this license or (at your option) any later
%% version.  The latest version of this license is in:
%% 
%%     http://www.latex-project.org/lppl.txt
%% 
%% and version 1.2 or later is part of all distributions of
%% LaTeX version 1999/12/01 or later.
%% 
%% Currently this work has the LPPL maintenance status "maintained".
%% 
%% The Current Maintainer of this work is Vilar Camara Neto.
%% 
%% This work consists of the files ppgccufmg.dtx and
%% ppgccufmg.ins and the derived file ppgccufmg.cls.
%% 
\NeedsTeXFormat{LaTeX2e}[1999/12/01]
\ProvidesClass{ppgccufmg}
  [2011/04/25 v1.44 Classe para dissertacoes, teses e propostas de tese no
  formato PPGCC/ICEx/UFMG]


\RequirePackage{keyval}
\RequirePackage{setspace}
\RequirePackage{textpos}

\def\ppgcc@classname{ppgccufmg}

%% Opções gerais:

\newif\ifppgcc@phd                 % Tese de doutorado?
\newif\ifppgcc@msc                 % Dissertação de mestrado?
\newif\ifppgcc@proposal            % Opção de proposta/projeto?
\newif\ifppgcc@showcover           % Compor a falsa folha de rosto?
\newif\ifppgcc@showtitle           % Compor a folha de rosto?
\newif\ifppgcc@showfc              % Compor a Ficha Catalográfica?
\newif\ifppgcc@showapproval        % Incluir a Folha de Aprovação?
\newif\ifppgcc@showabstract        % Compor as páginas de resumo/abstract?
\newif\ifppgcc@showdedication      % Compor a dedicatória?
\newif\ifppgcc@showack             % Compor os agradecimentos?
\newif\ifppgcc@showepigraph        % Compor a Epígrafe?
\newif\ifppgcc@showlof             % Compor a Lista de Figuras?
\newif\ifppgcc@showlot             % Compor a Lista de Tabelas?
\newif\ifppgcc@showtoc             % Compor o Sumário?
\newif\ifppgcc@extraporttitlepage  % Exibir também uma folha de título
                                   % em português?
\newif\ifppgcc@putmainmatter       % Chamar automaticamente \mainmatter depois
                                   % das páginas introdutórias?

\newif\if@selfdoc
\ifx\ppgcc@selfdoc\undefined\else\@selfdoctrue\fi
\if@selfdoc\input{ppgccufmgselfdoc}\fi

\def\ppgcc@showall{%
  \ppgcc@showcovertrue
  \ppgcc@showtitletrue
  \ifppgcc@proposal\ppgcc@showfcfalse\else\ppgcc@showfctrue\fi
  \ppgcc@showapprovaltrue
  \ppgcc@showabstracttrue
  \ppgcc@showdedicationtrue
  \ppgcc@showacktrue
  \ppgcc@showepigraphtrue
  \ppgcc@showloftrue
  \ppgcc@showlottrue
  \ppgcc@showtoctrue
}

\def\ppgcc@hideall{%
  \ppgcc@showcoverfalse
  \ppgcc@showtitlefalse
  \ppgcc@showfcfalse
  \ppgcc@showapprovalfalse
  \ppgcc@showabstractfalse
  \ppgcc@showdedicationfalse
  \ppgcc@showackfalse
  \ppgcc@showepigraphfalse
  \ppgcc@showloffalse
  \ppgcc@showlotfalse
  \ppgcc@showtocfalse
}

\ppgcc@proposalfalse
\ppgcc@showall
\ppgcc@putmainmattertrue

%% Opções das referências bibliográficas:

\newif\if@bibnorepauthor  % Substituir autorias repetidas por linha comprida?
\newif\if@nobreakitems    % Impedir quebra de página no meio de um item?

\@bibnorepauthorfalse
\@nobreakitemsfalse

\def\ppgcc@gobblespaces #1{#1}%

\def\BreakableUppercase#1{%
  \bgroup
  \let\ppgcc@prevdbs=\\%
  \def\\{\protect\ppgcc@prevdbs}%
  \MakeUppercase{#1}%
  \egroup
}

\newcommand{\ppgcc@defspacing}{\onehalfspace}

\def\ppgcc@redefchaptitlefont{}

\newcommand*{\authorrev}[1]{\gdef\ppgcc@authorrev{#1}} % Nome rev. do autor
\newcommand*{\advisor}[1]{\gdef\ppgcc@advisorname{#1}} % Nome do orientador
\newcommand*{\university}[1]{\gdef\@university{#1}}    % Nome da universidade
\newcommand*{\course}[1]{\gdef\@course{#1}}            % Nome do curso
\newcommand*{\address}[1]{\gdef\@address{#1}}          % Endereço (cidade/UF)
\newcommand*{\cutter}[1]{\gdef\ppgcc@cutter{#1}}       % Código "cutter"
\newcommand*{\cdu}[1]{\gdef\ppgcc@cdu{#1}}             % Código CDU
\newcommand*{\indexkeys}[1]{\gdef\ppgcc@indexkeys{#1}} % Chaves para indexação

\let\ppgcc@origdatecmd=\date
\renewcommand{\date}[1]{%                              % Data (aaaa-mm ou aaaa-mm-dd)
  \ppgcc@origdatecmd{#1}\ppgcc@parsedate#1---\relax}

\newcommand*{\coursedegree}[1]{\gdef\@degree{#1}}      % Titulação
\renewcommand*{\title}[1]                             % Título
  {\gdef\@title{#1}}

\newcommand*{\portuguesetitle}[1]                     % Título em português
  {\gdef\@portuguesetitle{#1}}
\newcommand*{\portugueseuniversity}[1]                % Nome da univers. em pt.
  {\gdef\@portugueseuniversity{#1}}
\newcommand*{\portuguesecourse}[1]                    % Nome do curso em port.
  {\gdef\@portuguesecourse{#1}}

\newcounter{ppgcc@c@keywords}

\def\ppgcc@keywords#1,#2\@{%
  \def\ppgcc@temp{#1}%
  \ifx\ppgcc@temp\empty\else
    \edef\ppgcc@temp{%
      \arabic{ppgcc@c@keywords}.\noexpand~\ppgcc@gobblespaces #1. }%
    \expandafter\expandafter\expandafter\def
      \expandafter\expandafter\expandafter\ppgcc@fc@keywords
      \expandafter\expandafter\expandafter{%
      \expandafter\ppgcc@fc@keywords\ppgcc@temp}%
    \stepcounter{ppgcc@c@keywords}%
    \expandafter\ppgcc@keywords#2\@
  \fi
}

\newcommand{\fckeywords}[1]{%
  \setcounter{ppgcc@c@keywords}{1}%
  \gdef\ppgcc@fc@keywords{}%
  \ppgcc@keywords#1,,\@
}
\newlength{\ppgcc@approvalwidth}
\newlength{\ppgcc@approvalvertadjust}

\newcommand*{\approval@opt}[1][0pt]{%
  \setlength{\ppgcc@approvalvertadjust}{#1}%
  \approval@opt@}
\newcommand{\approval@opt@}[1][1]{%
  \setlength{\ppgcc@approvalwidth}{#1\textwidth}%
  \approval@opt@@}
\def\approval@opt@@#1\@{%
  \gdef\ppgcc@approval{#1}}                           % Arquivo com a F.A.
\newcommand*{\approval}[1][0pt]{%
  \setlength{\ppgcc@approvalvertadjust}{#1}%
  \approval@}
\newcommand{\approval@}[2][1]{%
  \setlength{\ppgcc@approvalwidth}{#1\textwidth}%
  \gdef\ppgcc@approval{#2}}                           % Arquivo com a F.A.
\newcommand*{\epigraphtext}[2]{%
  \gdef\ppgcc@epigraphtext{#1}\gdef\ppgcc@epigraphauthor{#2}%
}

\DeclareOption{msc}{\ppgcc@phdfalse\ppgcc@msctrue}
\DeclareOption{phd}{\ppgcc@phdtrue\ppgcc@mscfalse}
\DeclareOption{proposal}{%
  \ppgcc@proposaltrue\ppgcc@showfcfalse\ppgcc@showapprovalfalse}
\DeclareOption{project}{%
  \ppgcc@proposaltrue\ppgcc@showfcfalse\ppgcc@showapprovalfalse}

\DeclareOption{single}{\renewcommand{\ppgcc@defspacing}{\singlespace}}
\DeclareOption{onehalf}{\renewcommand{\ppgcc@defspacing}{\onehalfspace}}
\DeclareOption{double}{\renewcommand{\ppgcc@defspacing}{\doublespace}}

\DeclareOption{hideall}{\ppgcc@hideall}
\DeclareOption{hidecover}{\ppgcc@showcoverfalse}
\DeclareOption{hidetitle}{\ppgcc@showtitlefalse}
\DeclareOption{hidefc}{\ppgcc@showfcfalse}
\DeclareOption{hideapproval}{\ppgcc@showapprovalfalse}
\DeclareOption{hideabstract}{\ppgcc@showabstractfalse}
\DeclareOption{hidededication}{\ppgcc@showdedicationfalse}
\DeclareOption{hideack}{\ppgcc@showackfalse}
\DeclareOption{hideepigraph}{\ppgcc@showepigraphfalse}
\DeclareOption{hidelof}{\ppgcc@showloffalse}
\DeclareOption{hidelot}{\ppgcc@showlotfalse}
\DeclareOption{hidetoc}{\ppgcc@showtocfalse}

\DeclareOption{showall}{\ppgcc@showall}
\DeclareOption{showcover}{\ppgcc@showcovertrue}
\DeclareOption{showtitle}{\ppgcc@showtitletrue}
\DeclareOption{showfc}{\ppgcc@showfctrue}
\DeclareOption{showapproval}{\ppgcc@showapprovaltrue}
\DeclareOption{showabstract}{\ppgcc@showabstracttrue}
\DeclareOption{showdedication}{\ppgcc@showdedicationtrue}
\DeclareOption{showack}{\ppgcc@showacktrue}
\DeclareOption{showepigraph}{\ppgcc@showepigraphtrue}
\DeclareOption{showlof}{\ppgcc@showloftrue}
\DeclareOption{showlot}{\ppgcc@showlottrue}
\DeclareOption{showtoc}{\ppgcc@showtoctrue}

\DeclareOption{a4paper}{\def\ppgcc@papersize{a4paper}}
\DeclareOption{letterpaper}{\def\ppgcc@papersize{letterpaper}}

\DeclareOption{9pt}{\def\ppgcc@ptsize{9pt}}
\DeclareOption{10pt}{\def\ppgcc@ptsize{10pt}}
\DeclareOption{11pt}{\def\ppgcc@ptsize{11pt}}
\DeclareOption{12pt}{\def\ppgcc@ptsize{12pt}}
\DeclareOption{14pt}{\def\ppgcc@ptsize{14pt}}
\DeclareOption{17pt}{\def\ppgcc@ptsize{17pt}}

\DeclareOption{nomainmatter}{\ppgcc@putmainmatterfalse}

\DeclareOption{centertitles}{%
   \def\ppgcc@redefchaptitlefont{%
    \expandafter\renewcommand\expandafter{\expandafter\chaptitlefont
    \expandafter}\expandafter{\chaptitlefont\centering}%
  }%
}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{memoir}}

\ExecuteOptions{a4paper,12pt,phd,onehalf,showall}
\ProcessOptions

\def\ppgcc@setdoctype{%
  \ifppgcc@phd
    % Esta é uma tese de doutorado
    \coursedegree{\ppgcc@phd}
    \ifppgcc@proposal
      \gdef\@documenttype{\ppgcc@phddocproj}%
      \ppgcc@set@phddocproj
    \else
      \gdef\@documenttype{\ppgcc@phddoc}%
      \ppgcc@set@phddoc
    \fi
  \else
    % Esta é uma dissertação de mestrado
    \coursedegree{\ppgcc@msc}
    \ifppgcc@proposal
      \gdef\@documenttype{\ppgcc@mscdocprop}%
      \ppgcc@set@mscdocprop
    \else
      \gdef\@documenttype{\ppgcc@mscdoc}%
      \ppgcc@set@mscdoc
    \fi
  \fi
}

\LoadClass[\ppgcc@papersize,\ppgcc@ptsize]{memoir}

\def\ppgcc@outermargin{2.5cm}
\def\ppgcc@innermargin{3.0cm}
\def\ppgcc@topmargin{2.0cm}
\def\ppgcc@bottommargin{2.0cm}
\def\ppgcc@toptextmargin{3.5cm}
\def\ppgcc@bottomtextmargin{3.0cm}
\def\ppgcc@headerheight{1.5cm}
\def\ppgcc@footerheight{1.0cm}

\setlrmarginsandblock{\ppgcc@innermargin}{\ppgcc@outermargin}{*}
\setulmarginsandblock{\ppgcc@toptextmargin}{\ppgcc@bottomtextmargin}{*}
\setheadfoot{\baselineskip}{\ppgcc@footerheight}
\setheaderspaces{\ppgcc@topmargin}{*}{*}
\checkandfixthelayout[fixed]

\maxsecnumdepth{subsubsection}
\setcounter{tocdepth}{2}
\setcounter{secnumdepth}{3}

\let\@author=\undefined


\def\ppgcc@@checkauthorrev#1,#2,#3,#4\@{%
  \def\ppgcc@temp{#2}\ifx\ppgcc@temp\empty
    \ClassError{\ppgcc@classname}{O texto de authorrev deve
      obrigatoriamente\MessageBreak
      ser fornecido no formato `Sobrenome, Nome'}{\@ehd}%
    \expandafter\endinput
  \fi
  \@ifundefined{@author}{%
    \edef\@author{\ppgcc@gobblespaces#2\space\ppgcc@gobblespaces#1}%
  }{}%
}
\def\ppgcc@checkauthor{%
  \ifx\@author\undefined
    \ifx\ppgcc@authorrev\undefined\else
      \expandafter\ppgcc@@checkauthorrev\ppgcc@authorrev,,,\@\relax
    \fi
  \fi
  \ifx\@author\undefined
    \ClassError{\ppgcc@classname}{O autor nao foi informado (use a chave
      authorrev ou author)}{\@ehd}%
    \expandafter\endinput
  \fi
}

%%%\ppgcc@checkauthorrev\ppgcc@authorrev,,,\@

\def\ppgcc@langgroup{\@ifundefined{languagename}{brazil}{\languagename}}

\def\ppgcc@deflanguage#1{%
  \@ifundefined{ppgcc@lang@#1}{%
    \ClassWarning{\ppgcc@classname}{Linguagem nao definida: #1^^J
    Adotando a linguagem padrao `brazil'.^^J}%
    \def\ppgcc@langgroup{brazil}}
    {\def\ppgcc@langgroup{#1}}%
}
\def\ppgcc@selectlanguage{%
  \csname ppgcc@lang@\ppgcc@langgroup\endcsname\ppgcc@setdoctype
}

\newif\ifppgcc@dateerror\ppgcc@dateerrortrue

\newcommand{\ppgcc@formatdate}[1][]{%
  \begingroup
    \def\ppgcc@templang{#1}%
    \ifx\ppgcc@templang\empty\let\ppgcc@templang\ppgcc@langgroup\fi
    \@ifundefined{ppgcc@year}{??}{\csname ppgcc@expanddate@\ppgcc@templang\endcsname}%
  \endgroup
}
\def\ppgcc@parsedate#1-#2-#3-#4\relax{%
  \ppgcc@dateerrortrue
  \gdef\ppgcc@year{\number#1}\gdef\ppgcc@month{\number#2}%
  \gdef\ppgcc@day{#3}\ifx\ppgcc@day\empty\else\gdef\ppgcc@day{\number#3}\fi
  \ifx\ppgcc@month\empty\else\ifnum#1>99\relax
    \ppgcc@dateerrorfalse
  \fi\fi
  \ifppgcc@dateerror
    \ClassError{\ppgcc@classname}{Formato invalido de data.^^J%
    Ao usar a opcao `date=' para o comando \string\ppgccufmg\space ou o
    comando \string\date\string{...\string},^^J%
    sempre forneca a data no formato aaaa-mm-dd ou aaaa-mm^^J%
    (note a ordem ano-mes-dia e o hifen como separador)}{\@ehd}%
    \expandafter\endinput
  \fi
}

\def\ppgcc@expandmonth@brazil#1{%
  \ifcase\ppgcc@month\or
  janeiro\or fevereiro\or mar\c{c}o\or abril\or maio\or junho\or
  julho\or agosto\or setembro\or outubro\or novembro\or dezembro\fi
}
\def\ppgcc@expanddate@brazil{%
  \edef\text@month{\ppgcc@expandmonth@brazil{\ppgcc@month}}%
  \ifx\ppgcc@day\empty\expandafter\MakeUppercase\text@month
    \else \two@digits\ppgcc@day\ de\ \text@month\fi
  \ de\ \ppgcc@year
}
\def\ppgcc@lang@brazil{%
  \ppgcc@extraporttitlepagefalse

  \def\@deflang@title{%
    \@ifundefined{@portuguesetitle}{\@title}{\@portuguesetitle}}
  \def\@deflang@university{%
    \@ifundefined{@portugueseuniversity}{\@university}%
    {\@portugueseuniversity}}
  \def\@deflang@course{%
    \@ifundefined{@portuguesecourse}{\@course}{\@portuguesecourse}}

  \def\ppgcc@msc{Mestre}
  \def\ppgcc@phd{Doutor}
  \def\ppgcc@mscdoc{Disserta\c{c}\~{a}o}
  \def\ppgcc@mscdocprop{Proposta de disserta\c{c}\~{a}o}
  \def\ppgcc@phddoc{Tese}
  \def\ppgcc@phddocproj{Projeto de tese}
  \def\ppgcc@advisor{Orientador}
  \def\ppgcc@docdescription{%
    \@documenttype{} apresentad\ppgcc@doctype@gender{} ao Programa de
    P\'{o}s\discretionary{-}{-}{-}Gradua\c{c}\~{a}o em \@deflang@course{} do
    Instituto de Ci\^{e}ncias Exatas da \@deflang@university{} como requisito
    parcial para a obten\c{c}\~{a}o do grau de \@degree{} em \@deflang@course.}
  \def\ppgcc@acknowledgments{Agradecimentos}
  \def\ppgcc@keywordsname{Palavras-chave}
  \def\ppgcc@indexkeys{\ppgcc@fc@keywords{} I.~Título.}

  \def\ppgcc@set@phddocproj{\def\ppgcc@doctype@gender{o}}
  \def\ppgcc@set@phddoc{\def\ppgcc@doctype@gender{a}}
  \def\ppgcc@set@mscdocprop{\def\ppgcc@doctype@gender{a}}
  \def\ppgcc@set@mscdoc{\def\ppgcc@doctype@gender{a}}

  \def\ppgccbibauthorsep{;}
  \def\ppgccbibauthorlastsep{}
  \def\ppgccbibauthorand{\&}
  \def\ppgccciteauthorand{\&}
  \def\ppgccbibinstring{Em}
  \def\ppgccbibpagestring{p.\@}
  \def\ppgccbibpagesstring{pp.\@}
  \def\ppgccbibeditorstring{editor}
  \def\ppgccbibeditorsstring{editores}
  \def\ppgccbibeditionstring{edi\c{c}\~{a}o}
  \def\ppgccbibchapterstring{cap\'{i}tulo}
  \def\ppgccbibtechreportstring{Relat\'{o}rio t\'{e}cnico}
  \def\ppgccbibmscstring{Disserta\c{c}\~{a}o de mestrado}
  \def\ppgccbibphdstring{Tese de doutorado}

  \def\ppgcc@appendixname{Ap\^{e}ndice}
  \def\ppgcc@attachmentname{Anexo}
}

\def\ppgcc@expandmonth@english#1{%
  \ifcase\ppgcc@month\or
  january\or february\or march\or april\or may\or june\or
  july\or august\or september\or october\or november\or december\fi
}
\def\ppgcc@expanddate@english{%
  \edef\text@month{\ppgcc@expandmonth@english{\ppgcc@month}}%
  \expandafter\MakeUppercase\text@month
  \ifx\ppgcc@day\empty\else\ \ppgcc@day,\fi
  \ \ppgcc@year
}
\def\ppgcc@lang@english{%
  \ppgcc@extraporttitlepagetrue

  \def\@deflang@title{\@title}
  \def\@deflang@university{\@university}
  \def\@deflang@course{\@course}

  \def\ppgcc@msc{Master}
  \def\ppgcc@phd{Doctor}
  \def\ppgcc@mscdoc{Dissertation}
  \def\ppgcc@mscdocprop{Dissertation proposal}
  \def\ppgcc@phddoc{Thesis}
  \def\ppgcc@phddocproj{Thesis project}
  \def\ppgcc@advisor{Advisor}
  \def\ppgcc@docdescription{%
    \@documenttype{} presented to the Graduate Program in \@course{} of the
    \@university{} in partial fulfillment of the requirements for the
    degree of \@degree{} in \@course.}
  \def\ppgcc@acknowledgments{Acknowledgments}
  \def\ppgcc@keywordsname{Keywords}
  \def\ppgcc@indexkeys{\ppgcc@fc@keywords{} I.~Title.}

  \let\ppgcc@set@phddocproj\relax
  \let\ppgcc@set@phddoc\relax
  \let\ppgcc@set@mscdocprop\relax
  \let\ppgcc@set@mscdoc\relax

  \def\ppgccbibauthorsep{,}
  \def\ppgccbibauthorlastsep{,}
  \def\ppgccbibauthorand{and}
  \def\ppgccciteauthorand{and}
  \def\ppgccbibinstring{In}
  \def\ppgccbibpagestring{page}
  \def\ppgccbibpagesstring{pages}
  \def\ppgccbibeditorstring{editor}
  \def\ppgccbibeditorsstring{editors}
  \def\ppgccbibeditionstring{edition}
  \def\ppgccbibchapterstring{chapter}
  \def\ppgccbibtechreportstring{Technical report}
  \def\ppgccbibmscstring{Master's thesis}
  \def\ppgccbibphdstring{PhD thesis}

  \def\ppgcc@appendixname{Appendix}
  \def\ppgcc@attachmentname{Attachment}
}
\newcommand{\makecoverpage}{%
  \ifppgcc@showcover\begingroup
    \ppgcc@deflanguage{brazil}\ppgcc@selectlanguage
    \ppgcc@maketitlepage{\z@}
  \endgroup\fi
}
\newcommand{\maketitlepage}{%
  \ifppgcc@showtitle
    \ifppgcc@extraporttitlepage\begingroup
      \ppgcc@deflanguage{brazil}\ppgcc@selectlanguage
      \ppgcc@maketitlepage{\@ne}%
    \endgroup\fi
    \ppgcc@maketitlepage{\@ne}%
  \fi
}

\def\ppgcc@docdescriptionwidth{0.5\textwidth}

\def\ppgcc@makedocdescription{%
  \begin{minipage}{\ppgcc@docdescriptionwidth}
    \ppgcc@docdescription
  \end{minipage}%
}

\def\ppgcc@titpag@fmtauthor#1{\large\BreakableUppercase{#1}}
\def\ppgcc@titpag@fmtadvisor{\large\scshape}
\def\ppgcc@titpag@fmttitle#1{\Large\bfseries\BreakableUppercase{#1}}
\def\ppgcc@titpag@fmtaddress{\large}
\def\ppgcc@titpag@fmtdate{\large}


\newenvironment{ppgcctitlingpageany}%
  {%
    \clearpage
    \thispagestyle{titlingpage}%
  }{%
    \thispagestyle{titlingpage}%
    \clearpage
  }

\newenvironment{ppgcctitlingpageodd}%
  {%
    \cleardoublepage
    \thispagestyle{titlingpage}%
  }{%
    \thispagestyle{titlingpage}%
    \clearpage
  }

\def\ppgcc@maketitlepageoptionalpart{%
  \ppgcc@makedocdescription\par
  \vspace{1.5cm}%
  {\ppgcc@titpag@fmtadvisor{\ppgcc@advisor: \ppgcc@advisorname}\par}%
}

\def\ppgcc@maketitlepage#1{%
  \ppgcc@checkauthor
  \begin{ppgcctitlingpageodd}
  \begin{singlespace}
  {\centering
    \vbox to3\baselineskip{%
      \ifx#1\z@
      \else
        {\ppgcc@titpag@fmtauthor{\@author}\par}%
        \vfil
      \fi
    }

    \vspace{\stretch{1}}

    \setbox0=\vbox{\ppgcc@maketitlepageoptionalpart}%
    \edef\ppgcc@textvspace{%
      \noexpand\vrule
      height\the\ht0
      depth\the\dp0
      width0pt\noexpand\par}
    \ppgcc@textvspace

    \vspace{1.5cm}

    \if@selfdoc
      \setbox0=\vbox{%
    \fi
    {\DoubleSpacing\ppgcc@titpag@fmttitle{\@deflang@title}\par}%
    \if@selfdoc
      }%
      \begin{tikzpicture}[overlay,remember picture]
        \node (tltitle) at (0,\ht0) {};
        \node (brtitle) at (\wd0,-\dp0) {};
      \end{tikzpicture}\box0
    \fi

    \vspace{1.5cm}

    \ifx#1\z@
      \ppgcc@textvspace
    \else
      \hspace*{\stretch{1}}%
      \if@selfdoc
        \setbox0=\hbox{%
      \fi
      \ppgcc@makedocdescription
      \if@selfdoc
        }%
        \begin{tikzpicture}[overlay,remember picture]
          \node (tldescr) at (0,\ht0) {};
          \node (brdescr) at (\wd0,-\dp0) {};
          \node (trdescr) at (tldescr -| brdescr) {};
          \node (bldescr) at (tldescr |- brdescr) {};
          %\draw [area] (trdescr) -- (tldescr) -- (bldescr) -- (brdescr);
          \draw [area] (trdescr) -- (tldescr -| left);
          \draw [area] (brdescr) -- (bldescr -| left);
          \draw [area] (tldescr) -- (bldescr);
          \node (bseparrow) at ([xshift=-1cm] tldescr) {};
          \draw [marrow] (bseparrow.center)
            -- node [left] {1,5cm} (bseparrow |- brtitle);
          \draw [marrow] (bldescr) ++(0,-.5cm)
            -- node [below] {Metade da largura}
            ($(brdescr) +(0,-.5cm)$);
          \draw [marrow] (bldescr) ++(-1cm,0)
            -- node [left] {1,5cm} ++(0,-1.5cm);
        \end{tikzpicture}%
        \box0\relax
      \fi

      \vspace{1.5cm}

      {\ppgcc@titpag@fmtadvisor{\ppgcc@advisor: \ppgcc@advisorname}\par}%
    \fi

    \vspace{\stretch{1}}

    \vbox to3\baselineskip{%
      \ifx#1\z@
      \else
        \vfil
        {\ppgcc@titpag@fmtaddress{\@address}\par}%
        \medskip
        {\ppgcc@titpag@fmtdate{\ppgcc@formatdate\par}}%
      \fi
    }\par
  }
  \if@selfdoc\global\let\ppgcc@selfdoc@fg\ppgcc@selfdoc@drawtitlepage\fi
  \end{singlespace}
  \end{ppgcctitlingpageodd}
}

\newif\if@fc@ilhookcalled\@fc@ilhookcalledfalse
\newif\if@fc@hasil\@fc@hasilfalse

\def\ppgcc@fc@ilaux{\global\@fc@hasiltrue}
\def\ppgcc@fc@ilhook{%
  \if@fc@ilhookcalled\else
    \global\@fc@ilhookcalledtrue
    \global\@fc@hasiltrue
    \immediate\write\@auxout{%
      \string\ppgcc@fc@ilaux{}^^J
    }%
    \typeout{* \ppgcc@classname: Pelo menos uma ilustracao encontrada.}%
  \fi
}

\newcommand{\ppgccaddilcmd}[1]{%
  \begingroup
  \let\ppgcc@temp#1
  \expandafter\gdef\expandafter#1\expandafter{%
    \expandafter\ppgcc@fc@ilhook\ppgcc@temp}%
  \endgroup
}
\newcommand{\ppgccaddilenv}[1]{%
  \expandafter\ppgccaddilcmd\expandafter{\csname#1\endcsname}%
}

\AtBeginDocument{\ppgccaddilenv{figure}}

\def\ppgcc@fc@width{12.5cm}

\def\ppgcc@fc@sep{1em}
\def\ppgcc@fc@rule{.8pt}

\def\ppgcc@fc@smallsep{0cm}
\def\ppgcc@fc@bigsep{1em}

\def\ppgcc@fc@parindent{.5cm}
\def\ppgcc@fc@colsep{1em}

\newcounter{ppgcc@fc@frontpages}

\let\ppgcc@mainmatter=\mainmatter
\def\mainmatter{
  \clearpage
  \setcounter{ppgcc@fc@frontpages}{\c@page}%
  \addtocounter{ppgcc@fc@frontpages}{-1}%
  \immediate\write\@auxout{%
    \string\setcounter{ppgcc@fc@frontpages}{\the\c@ppgcc@fc@frontpages}^^J
  }%
  \ppgcc@mainmatter
}

\def\ppgcc@fc@checkerr#1#2#3{%
  \ClassError{\ppgcc@classname}{%
    Para a montagem da Ficha Catalografica e' necessario\MessageBreak
    definir #1 (chave `#2'\MessageBreak
    ou comando \noexpand#3)}{\@ehd}
    \endinput
}
\def\ppgcc@fc@checkwarn#1#2#3{%
  \ClassWarning{\ppgcc@classname}{%
    Para a montagem da Ficha Catalografica e' necessario\MessageBreak
    definir #1 (chave `#2'\MessageBreak
    ou comando \noexpand#3)}
}
\def\ppgcc@fc@checkparamerr#1#2#3#4{%
  \@ifundefined{#1}
    {\ppgcc@fc@checkerr{#2}{#3}{#4}}{}
}
\def\ppgcc@fc@checkparamwarn#1#2#3#4{%
  \@ifundefined{#1}
    {\ppgcc@fc@checkwarn{#2}{#3}{#4}}{}
}
\def\ppgcc@fc@check{%
  \ppgcc@fc@checkparamerr{ppgcc@authorrev}{o nome reverso do autor}{authorrev}
    {\authorrev}
  \ppgcc@fc@checkparamwarn{ppgcc@cutter}{o código ``cutter''}{cutter}{\cutter}
  \ppgcc@fc@checkparamwarn{ppgcc@cdu}{o código CDU}{cdu}{\cdu}
}

\def\ppgcc@titlenobreak{%
  \begingroup
    \let\\=\space
    \@title
  \endgroup
}

\newlength{\ppgcc@l@fc@width}
\newlength{\ppgcc@l@fc@seccolwidth}
\newlength{\ppgcc@l@fc@firstcolwidth}

\def\ppgcc@fc@calc{%
  \ppgcc@fc@check

  \@ifundefined{ppgcc@cutter}{%
    \setlength{\ppgcc@l@fc@firstcolwidth}{0pt}
    \def\ppgcc@fc@realcolsep{0pt}
  }{%
    \settowidth{\ppgcc@l@fc@firstcolwidth}{\ppgcc@cutter}
    \let\ppgcc@fc@realcolsep=\ppgcc@fc@colsep
  }

  \setlength{\ppgcc@l@fc@width}{\ppgcc@fc@width}
  \addtolength{\ppgcc@l@fc@width}{-\ppgcc@fc@rule}
  \addtolength{\ppgcc@l@fc@width}{-\ppgcc@fc@rule}
  \addtolength{\ppgcc@l@fc@width}{-\ppgcc@fc@sep}
  \addtolength{\ppgcc@l@fc@width}{-\ppgcc@fc@sep}

  \setlength{\ppgcc@l@fc@seccolwidth}{\ppgcc@l@fc@width}
  \addtolength{\ppgcc@l@fc@seccolwidth}{-\ppgcc@l@fc@firstcolwidth}
  \addtolength{\ppgcc@l@fc@seccolwidth}{-\ppgcc@fc@realcolsep}
}

\def\ppgcc@makefc{%
  \begin{ppgcctitlingpageany}%
  \begin{singlespace}%

  \ppgcc@checkauthor
  \ppgcc@fc@calc

  \hyphenpenalty=10000
  \flushbottom

  \noindent
  \begin{tabular}{@{}ll@{}}
    \copyright & \ppgcc@year, \@author. \\
    & Todos os direitos reservados.
  \end{tabular}

  \leavevmode\vfill

  \fboxsep=\ppgcc@fc@sep
  \fboxrule=\ppgcc@fc@rule
  \centering\noindent
  \if@selfdoc
    \setbox0=\hbox{%
  \fi
  \fbox{%
  \begin{minipage}{\ppgcc@l@fc@width}
    \begin{tabular}%
      {@{}p{\ppgcc@l@fc@firstcolwidth}%
      @{\hspace{\ppgcc@fc@realcolsep}}%
      >{\raggedright\arraybackslash}%
      p{\ppgcc@l@fc@seccolwidth}@{}}
      & \ppgcc@authorrev
        \\[\ppgcc@fc@smallsep]
      \@ifundefined{ppgcc@cutter}{}{\ppgcc@cutter}%
      & \hspace*{\ppgcc@fc@parindent}%
        \ppgcc@titlenobreak{} / \@author.\space --- \@address, \ppgcc@year
        \\[\ppgcc@fc@smallsep]
      & \hspace*{\ppgcc@fc@parindent}%
        \roman{ppgcc@fc@frontpages}, \arabic{lastpage} f.~%
        \if@fc@hasil : il.~\fi
        ; 29cm
        \\[\ppgcc@fc@bigsep]
      & \hspace*{\ppgcc@fc@parindent}%
        \ifppgcc@phd
          Tese (doutorado)
        \else
          Disserta\c{c}\~{a}o (mestrado)
        \fi
        --- \@university
        \\[\ppgcc@fc@smallsep]
      & \hspace*{\ppgcc@fc@parindent}%
        Orientador: \ppgcc@advisorname
        \\[\ppgcc@fc@bigsep]
      & \hspace*{\ppgcc@fc@parindent}%
        \ppgcc@indexkeys
    \end{tabular}
    \@ifundefined{ppgcc@cdu}{}{%
      \vspace{\ppgcc@fc@bigsep}\par
      {\centering
        CDU \ppgcc@cdu
      \par}%
    }%
  \end{minipage}}%
  \if@selfdoc
    }\begingroup
      \edef\@ht{\the\ht0}\edef\@dp{\the\dp0}%
      \box0\relax
      \ppgcc@selfdoc@begin
      \ppgcc@selfdoc@drawfc
      \ppgcc@selfdoc@end
    \endgroup
  \fi
  \end{singlespace}
  \end{ppgcctitlingpageany}
  \clearpage
}

\newcommand{\makefc}{%
  \ifppgcc@showfc\ppgcc@makefc\fi
}

\newcommand{\makeapproval}{%
  \ifppgcc@showapproval
    \begin{ppgcctitlingpageodd}
    \begin{singlespace}

    \ifx\ppgcc@approval\undefined
      \leavevmode\vfil
      {\centering
        {\huge [Folha de Aprova\c{c}\~ao]\par}
        \bigskip
        Quando a secretaria do Curso fornecer esta folha,\par
        ela deve ser digitalizada e armazenada no disco em formato
          gr\'afico.\par
        \medskip
        Se voc\^e estiver usando o \texttt{pdflatex},\par
        armazene o arquivo preferencialmente em formato PNG\par
        (o formato JPEG \'e pior neste caso).\par
        \medskip
        Se voc\^e estiver usando o \texttt{latex} (n\~ao o
          \texttt{pdflatex}),\par
        ter\'a que converter o arquivo gr\'afico para o formato EPS.\par
        \medskip
        Em seguida, acrescente a op\c{c}\~ao
        \texttt{approval=\{}\emph{nome do arquivo}\texttt{\}}\par
        ao comando \texttt{\textbackslash ppgccufmg}.\par
        \medskip
        Se a imagem da folha de aprova\c{c}\~ao precisar ser ajustada, use:\par
        \texttt{approval=[}\emph{ajuste}\texttt{][}\emph{escala}\texttt{]\{}%
        \emph{nome do arquivo}\texttt{\}}\par
        onde \emph{ajuste} é uma dist\^ancia para deslocar a imagem para baixo\par
        e \emph{escala} \'e um fator de escala para a imagem. Por exemplo:\par
        \texttt{approval=[-2cm][0.9]\{}\emph{nome do arquivo}\texttt{\}}\par
        desloca a imagem 2cm para cima e a escala em 90\%.\par
      }
    \else
      \begingroup
        \setlength{\TPHorizModule}{\ppgcc@approvalwidth}%
        \begin{textblock}{1}(0,0)
          \vspace*{\ppgcc@approvalvertadjust}%
          \noindent\includegraphics[width=\ppgcc@approvalwidth]{\ppgcc@approval}
        \end{textblock}
      \endgroup
    \fi

    \end{singlespace}
    \end{ppgcctitlingpageodd}
  \fi
}

\newcommand{\includeabstract}[3][]{%
  \ifppgcc@showabstract
    \cleardoublepage
    \begingroup
      \def\ppgcc@temp{#1}%
      \ifx\ppgcc@temp\empty\else
        \ppgcc@deflanguage{#1}\ppgcc@selectlanguage
      \fi
      \ppgcc@redefchaptitlefont
      \chapter{#2}
      \input{#3}
    \endgroup
    \clearpage
  \fi
}

\newcounter{ppgcc@c@abstract}

\newcommand{\addabstract}[3][]{%
  \stepcounter{ppgcc@c@abstract}%
  \expandafter\gdef
    \csname ppgcc@abstract@\theppgcc@c@abstract lang\endcsname{#1}%
  \expandafter\gdef
    \csname ppgcc@abstract@\theppgcc@c@abstract title\endcsname{#2}%
  \expandafter\gdef
    \csname ppgcc@abstract@\theppgcc@c@abstract filename\endcsname{#3}%
}

\newcommand{\ppgcc@includeabstractlist}{%
  \ifppgcc@showabstract
    \setcounter{ppgcc@c@abstract}{1}
    \loop\expandafter\ifx
      \csname ppgcc@abstract@\theppgcc@c@abstract title\endcsname
      \relax
      \else
      \includeabstract
        [\csname ppgcc@abstract@\theppgcc@c@abstract lang\endcsname]
        {\csname ppgcc@abstract@\theppgcc@c@abstract title\endcsname}
        {\csname ppgcc@abstract@\theppgcc@c@abstract filename\endcsname}
      \stepcounter{ppgcc@c@abstract}
    \repeat
  \fi
}

\newcommand{\keywords}[1]{%
  \bigskip
  \noindent\textbf{\ppgcc@keywordsname:} #1.%
}

\newcounter{ppgcc@c@dedication}
\newcounter{ppgcc@c@ack}

\newcommand{\adddedication}[1]{%
  \stepcounter{ppgcc@c@dedication}
  \expandafter\def\csname ppgcc@dedication@\theppgcc@c@dedication
    filename\endcsname{#1}
}

\newcommand{\addack}[2][]{%
  \stepcounter{ppgcc@c@ack}
  \def\ppgcc@tempack{#1}
  \ifx\ppgcc@tempack\empty
    \expandafter\addack@\expandafter{\ppgcc@acknowledgments}
  \else
    \addack@{#1}
  \fi
  \expandafter\def\csname ppgcc@ack@\theppgcc@c@ack filename\endcsname{#2}
}
\def\addack@#1{
  \expandafter\def\csname ppgcc@ack@\theppgcc@c@ack title\endcsname{#1}
}

\newcommand{\includededication}[1]{%
  \ifppgcc@showdedication
    \cleardoublepage
    \vspace*{\stretch{1}}
    \textit{\input{#1}}
    \vspace*{\stretch{1}}
    \clearpage
  \fi
}

\newcommand{\includeack}[2][]{%
  \ifppgcc@showack
    \edef\ppgcc@tempack{#1}%
    \ifx\ppgcc@tempack\empty\let\ppgcc@tempack\ppgcc@acknowledgments\fi
    \ppgcc@redefchaptitlefont
    \chapter{\ppgcc@tempack}
    \input{#2}
    \clearpage
  \fi
}

\newcommand{\ppgcc@includededicationlist}{%
  \ifppgcc@showdedication
    \setcounter{ppgcc@c@dedication}{1}
    \loop
    \expandafter\ifx
      \csname ppgcc@dedication@\theppgcc@c@dedication filename\endcsname
      \relax
    \else
      \includededication
        {\csname ppgcc@dedication@\theppgcc@c@dedication filename\endcsname}
      \stepcounter{ppgcc@c@dedication}
    \repeat
  \fi
}
\newcommand{\ppgcc@includeacklist}{%
  \ifppgcc@showack
    \setcounter{ppgcc@c@ack}{1}
    \loop
    \expandafter\ifx\csname ppgcc@ack@\theppgcc@c@ack filename\endcsname\relax
    \else
      \includeack
        [\csname ppgcc@ack@\theppgcc@c@ack title\endcsname]
        {\csname ppgcc@ack@\theppgcc@c@ack filename\endcsname}
      \stepcounter{ppgcc@c@ack}
    \repeat
  \fi
}

\def\ppgcc@epigraph@fmttext#1{\textit{``#1''}}
\def\ppgcc@epigraph@fmtauthor#1{(#1)}
\def\ppgcc@epigraph@fmtvspace{0em}

\newcommand{\makeepigraph}{%
  \ifppgcc@showepigraph
    \@ifundefined{ppgcc@epigraphtext}{}{%
      \cleardoublepage
      \vspace*{\stretch{1}}
      \begingroup
        \raggedleft
        {\ppgcc@epigraph@fmttext{\ppgcc@epigraphtext}\par}%
        \vspace{\ppgcc@epigraph@fmtvspace}
        {\ppgcc@epigraph@fmtauthor{\ppgcc@epigraphauthor}\par}%
      \endgroup
      \clearpage
    }%
  \fi
}

\if@twoside
  \makeheadrule{headings}{\textwidth}{0pt}
  \makeoddhead{headings}{\textsc{\rightmark}}{}{\thepage}
  \makeevenhead{headings}{\thepage}{}{\textsc{\leftmark}}

  \copypagestyle{contents}{plain}

  \copypagestyle{listoffigures}{plain}

  \copypagestyle{listoftables}{plain}

  \copypagestyle{bibliography}{headings}
  \makeoddhead{bibliography}{\textsc{\bibname}}{}{\thepage}
  \makeevenhead{bibliography}{\thepage}{}{\textsc{\bibname}}
\else
  \makeheadrule{headings}{\textwidth}{0pt}
  \makeoddhead{headings}{\textsc{\leftmark}}{}{\thepage}

  \copypagestyle{contents}{plain}

  \copypagestyle{listoffigures}{plain}

  \copypagestyle{listoftables}{plain}

  \copypagestyle{bibliography}{headings}
  \makeoddhead{bibliography}{\textsc{\bibname}}{}{\thepage}
\fi

\def\ppgcc@chaptermark#1{%
  \markboth{%
    \ifnum\c@secnumdepth>\m@ne
      \if@mainmatter
        \if@twoside\@chapapp\ \fi
        \thechapter.
      \fi
    \fi
    #1}{}}%

\def\ppgcc@sectionmark#1{%
  \markright{%
    \ifnum\c@secnumdepth>\z@\thesection. \ \fi
    #1}}%

\def\ppgccrestoremarks{%
  \let\chaptermark=\ppgcc@chaptermark
  \let\sectionmark=\ppgcc@sectionmark
}

\ppgccrestoremarks
\g@addto@macro\mainmatter{\pagestyle{headings}\ppgccrestoremarks}

\let\ppgcc@orig@listoffigures=\listoffigures
\renewcommand{\listoffigures}{{%
  \ifppgcc@showlof\begingroup
    \cleardoublepage
    \ppgcc@redefchaptitlefont
    \let\ppgcc@save@listoffigures=\listoffigures
    \let\listoffigures=\ppgcc@orig@listoffigures
    \pagestyle{listoffigures}
    \listoffigures
    \clearpage
  \endgroup\fi
}}

\let\ppgcc@orig@listoftables=\listoftables
\renewcommand{\listoftables}{{%
  \ifppgcc@showlot\begingroup
    \cleardoublepage
    \ppgcc@redefchaptitlefont
    \let\ppgcc@save@listoftables=\listoftables
    \let\listoftables=\ppgcc@orig@listoftables
    \pagestyle{listoftables}
    \listoftables
    \clearpage
  \endgroup\fi
}}

\let\ppgcc@orig@tableofcontents=\tableofcontents
\renewcommand{\tableofcontents}{{%
  \ifppgcc@showtoc\begingroup
    \cleardoublepage
    \ppgcc@redefchaptitlefont
    \let\ppgcc@save@tableofcontents=\tableofcontents
    \let\tableofcontents=\ppgcc@orig@tableofcontents
    \pagestyle{contents}
    \tableofcontents*
    \clearpage
  \endgroup\fi
}}


\let\ppgcc@beforetoc\empty

\newcommand\ppgcc@addabstract[3][]{\addabstract[#1]{#2}{#3}}
\def\ppgcc@adddedication#1{\adddedication{#1}}
\def\ppgcc@addack{\@ifnextchar[{\addack}{\ppgcc@addack@}}
\def\ppgcc@addack@#1\relax{\addack[]{#1}}
\def\ppgcc@addbeforetoc#1{\g@addto@macro\ppgcc@beforetoc{#1}}

\define@key{ppgcc}{title}{\title{#1}}
\define@key{ppgcc}{author}{\author{#1}}
\define@key{ppgcc}{authorrev}{\authorrev{#1}}
\define@key{ppgcc}{cutter}{\cutter{#1}}
\define@key{ppgcc}{cdu}{\cdu{#1}}
\define@key{ppgcc}{indexkeys}{\indexkeys{#1}}
\define@key{ppgcc}{university}{\university{#1}}
\define@key{ppgcc}{course}{\course{#1}}
\define@key{ppgcc}{portuguesetitle}{\portuguesetitle{#1}}
\define@key{ppgcc}{portugueseuniversity}{\portugueseuniversity{#1}}
\define@key{ppgcc}{portuguesecourse}{\portuguesecourse{#1}}
\define@key{ppgcc}{keywords}{\fckeywords{#1}}
\define@key{ppgcc}{address}{\address{#1}}
\define@key{ppgcc}{date}{\date{#1}}
\define@key{ppgcc}{approval}{\approval@opt#1\@}
\define@key{ppgcc}{epigraphtext}{\epigraphtext#1}
\define@key{ppgcc}{advisor}{\advisor{#1}}
\define@key{ppgcc}{abstract}{\ppgcc@addabstract#1}
\define@key{ppgcc}{dedication}{\ppgcc@adddedication{#1}}
\define@key{ppgcc}{ack}{\ppgcc@addack#1\relax}
\define@key{ppgcc}{beforetoc}{\ppgcc@addbeforetoc{#1}}

\newcommand{\ppgccufmg}[1]{%
  \setkeys{ppgcc}{#1}

  \pagestyle{plain}

  \frontmatter

  % Falsa folha de rosto
  \makecoverpage
  % Folha(s) de rosto
  \maketitlepage
  % Ficha Catalográfica
  \makefc
  % Folha de aprovação
  \makeapproval
  % Dedicatória
  \ppgcc@includededicationlist
  % Agradecimentos
  \ppgcc@includeacklist
  % Epígrafe
  \makeepigraph
  % Resumo, Abstract, etc.
  \ppgcc@includeabstractlist

  % Sumário e lista de figuras e de tabelas
  \listoffigures
  \listoftables
  \ppgcc@beforetoc
  \tableofcontents

  % ***** ATENÇÃO *****
  %
  % Se precisar incluir outras listas, como a Lista de Algoritmos ou de
  % Símbolos, NÃO ALTERE ESTE ARQUIVO PARA COLOCAR O COMANDO AQUI, senão o
  % documento será gerado com problemas! Veja o exemplo da seção "Para os
  % impacientes", que descreve esse caso.

  \ifppgcc@putmainmatter\mainmatter\fi
}

\def\ppgcc@bibauthorformat{}
\def\ppgcc@bibtitleformat{}
\def\ppgcc@bibbtitleformat{\itshape}
\def\ppgcc@bibbooktitleformat{\itshape}
\def\ppgcc@bibjournalformat{\itshape}

\def\ppgcc@norepstring{---------}

\newcommand{\ppgccbibauthordoformat}[1]{%
  \if@bibnorepauthor
    \def\ppgcc@currentauthor{#1}%
    \ifx\ppgcc@lastauthor\ppgcc@currentauthor\ppgcc@norepstring
      \else{\ppgcc@bibauthorformat#1}\fi
    \def\ppgcc@lastauthor{#1}%
  \else
    {\ppgcc@bibauthorformat#1}%
  \fi
}
\newcommand{\ppgccbibtitledoformat}[1]{{\ppgcc@bibtitleformat#1}}
\newcommand{\ppgccbibbtitledoformat}[1]{{\ppgcc@bibbtitleformat#1}}
\newcommand{\ppgccbibbooktitledoformat}[1]{{\ppgcc@bibbooktitleformat#1}}
\newcommand{\ppgccbibjournaldoformat}[1]{{\ppgcc@bibjournalformat#1}}
\let\ppgccyeardoformat\empty
\let\ppgccpagedoformat\empty
\let\ppgccnumberdoformat\empty
\let\ppgccvolumedoformat\empty

\define@key{ufmgbib}{noauthorrepeat}[true]{\@bibnorepauthortrue}
\define@key{ufmgbib}{noauthorrepstring}{\def\ppgcc@norepstring{#1}}
\define@key{ufmgbib}{nobreakitems}[true]{\@nobreakitemstrue}
\define@key{ufmgbib}{bibauthorand}{\def\ppgccbibauthorand{#1}}
\define@key{ufmgbib}{citeauthorand}{\def\ppgccciteauthorand{#1}}
\define@key{ufmgbib}{authorformat}{\def\ppgcc@bibauthorformat{#1}}
\define@key{ufmgbib}{titleformat}{\def\ppgcc@bibtitleformat{#1}}
\define@key{ufmgbib}{btitleformat}{\def\ppgcc@bibbtitleformat{#1}}
\define@key{ufmgbib}{booktitleformat}{\def\ppgcc@bibbooktitleformat{#1}}
\define@key{ufmgbib}{journalformat}{\def\ppgcc@bibjournalformat{#1}}

\newcommand{\ppgccbibliography}[2][]{%
  \setkeys{ufmgbib}{#1}

  \cleardoublepage
  \pagestyle{bibliography}
  \bibliographystyle{ppgccufmg}
  \begingroup
    \ppgcc@redefchaptitlefont
    \if@nobreakitems\raggedbottom\interlinepenalty=10000\relax\fi
    \bibliography{#2}
  \endgroup

  \clearpage
  \pagestyle{headings}
  \ppgccrestoremarks
}

\def\ppgcc@toc@switchto#1{%
  \ifx\ppgcc@chapternumberline\undefined
    \let\ppgcc@chapternumberline\chapternumberline
  \fi
  \def\chapternumberline{#1\space\ppgcc@chapternumberline}%
}
\def\ppgcc@toc@recover{%
  \let\chapternumberline\ppgcc@chapternumberline
  \let\ppgcc@chapternumberline\undefined
}

\let\ppgcc@appendices\appendices\let\appendices\undefined
\let\ppgcc@endappendices\endappendices\let\endappendices\undefined
\newenvironment{appendices}{%
    \cleardoublepage
    \ppgcc@appendices
    \appendix
    \let\appendixname\ppgcc@appendixname
    \immediate\write\@auxout{%
      \string\@writefile{toc}{%
      \string\ppgcc@toc@switchto\string\ppgcc@appendixname}}%
  }{%
    \ppgcc@endappendices
    \immediate\write\@auxout{%
      \string\@writefile{toc}{\string\ppgcc@toc@recover}}
  }

\newenvironment{attachments}{%
    \cleardoublepage
    \ppgcc@appendices
    \appendix
    \let\appendixname\ppgcc@attachmentname
    \def\Hy@appendixstring{attachment}%
    \immediate\write\@auxout{%
      \string\@writefile{toc}{%
      \string\ppgcc@toc@switchto\string\ppgcc@attachmentname}}%
    % Para o hyperref:
    \def\Hy@chapapp{attachment}%
  }{%
    \ppgcc@endappendices
    \immediate\write\@auxout{%
      \string\@writefile{toc}{\string\ppgcc@toc@recover}}
  }

\sloppy

\setlength{\parindent}{1cm}


\makechapterstyle{ppgccufmgsf}{%
  \chapterstyle{default}%
  \expandafter\def\expandafter\chapnumfont\expandafter{%
    \chapnumfont\sffamily}%
  \expandafter\def\expandafter\chapnamefont\expandafter{%
    \chapnamefont\sffamily}%
  \expandafter\def\expandafter\chaptitlefont\expandafter{%
    \chaptitlefont\sffamily}%
  \expandafter\def\expandafter\secheadstyle\expandafter{%
    \secheadstyle\sffamily}%
  \expandafter\def\expandafter\subsecheadstyle\expandafter{%
    \subsecheadstyle\sffamily}%
  \expandafter\def\expandafter\subsubsecheadstyle\expandafter{%
    \subsubsecheadstyle\sffamily}%
  \expandafter\def\expandafter\paraheadstyle\expandafter{%
    \paraheadstyle\sffamily}%
  \expandafter\def\expandafter\subparaheadstyle\expandafter{%
    \subparaheadstyle\sffamily}%
  \setlength{\beforechapskip}{20mm}%
  \setlength{\midchapskip}{7mm}%
  \setlength{\afterchapskip}{15mm}%
}
\chapterstyle{ppgccufmgsf}

\captionnamefont{\bfseries\small}
\captiondelim{. }
\captiontitlefont{\small}
\newlength{\ppgcc@captionwidth}
\setlength{\ppgcc@captionwidth}{\textwidth}
\addtolength{\ppgcc@captionwidth}{-2cm}
\captionwidth{\ppgcc@captionwidth}
\changecaptionwidth

\renewcommand*{\cftchapterformatpnum}[1]{%
  \hb@xt@\@pnumwidth{\hss\cftchapterpagefont #1}}

\AtBeginDocument{%
  \ppgcc@selectlanguage\ppgcc@defspacing
  \@ifpackageloaded{graphicx}{}{\RequirePackage{graphicx}}%
  \@ifpackageloaded{hyperref}{\RequirePackage{memhfixc}}{}%
}
\endinput
%%
%% End of file `ppgccufmg.cls'.
