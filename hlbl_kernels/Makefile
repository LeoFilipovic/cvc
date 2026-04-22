###########

# MODIFY DIRECTOR ACORDINGLY! 
work=/capstor/store/cscs/userlab/lp142/software/install_dir
srcdir=

limedir=$(work)/c-lime
lemondir=$(work)/other_lemon
lapackdir=$(work)/lapack
fftwdir=$(work)/fftw-2.1.5
lhpcdir=$(work)/aff
tmLQCD_src=$(work)/../src_dirs/tmLQCD
tmLQCDdir=$(work)/tmLQCD
tmLQCD_libs= -lwrapper -lhmc -lmonomial -loperator -lsolver -linit -lmeas -llinalg -lhmc -lxchange -lrational -lio -lsmear -lbuffers
qudadir=$(work)/quda/
kqedcudadir=$(work)/KQED/install_cuda/
kqeddir=$(work)/KQED/install_cpu/
cudadir=${CUDA_HOME}


defines= -DF_ -DHAVE_MPI -DPARALLELTXYZ -DHAVE_LIBLEMON -DHAVE_LHPC_AFF -DHAVE_TMLQCD_LIBWRAPPER -DTM_USE_MPI -DHAVE_OPENMP -DHAVE_HDF5 -DTM_USE_QUDA -D_SMEAR_QUDA -DHAVE_KQED
defines+=-DGPU_DIRECT_SOLVER
# defines+=-DDUMMY_SOLVER
gpu_defines=-DHAVE_CUDA -DUSE_CUDA


CXX=mpicxx
CXXFLAGS= -fopenmp -pedantic -Wall -g -O3 -mcpu=native $(defines)
CXXFLAGS_GPU= -fopenmp -Wall -g -O3 -mcpu=native $(defines) $(gpu_defines)
NVCC=nvcc -ccbin=$(CXX) #-lineinfo
NVCCFLAGS= -arch=sm_90 --ptxas-options=-v
# -DHAVE_OPENMP
# -DCUBIC_GROUP_DOUBLE_COVER


CCDEP = g++
DEPFLAGS = -MM

INCLUDE = -I. -I$(srcdir) \
	-I$(fftwdir)/include/ -I$(limedir)/include/ \
	-I$(lemondir)/include/ \
	-I$(lhpcdir)/include/ \
	-I$(tmLQCDdir)/include/ -I$(tmLQCD_src)/include/ \
	$(shell pkg-config hdf5 --cflags) \
	-I$(lapackdir)/include
INCLUDE_CPU=$(INCLUDE) -I$(kqeddir)/include/
INCLUDE_GPU=$(INCLUDE) -I$(kqedcudadir)/include/ -I$(cudadir)/include/ 

LIBS = -lm \
	-L$(fftwdir)/lib/ -lfftw -lfftw_mpi \
	-L$(tmLQCDdir)/lib/ $(tmLQCD_libs) \
	-L$(lemondir)/lib/ -llemon \
	-L$(limedir)/lib/ -llime \
	-L$(lhpcdir)/lib/ -llhpc-aff \
	-L$(qudadir)/lib/ -lquda \
	-L$(cudadir)/lib64/ -lcuda -lcudart \
	$(shell pkg-config hdf5 --libs) \
	-L$(lapackdir)/lib64/ -llapack -lblas -lgfortran
LIB_KQED_GPU=-L$(kqedcudadir)/lib/ -lKQED_cuda
LIB_KQED=-L$(kqeddir)/lib/ -lKQED
LIBS_GPU=$(LIBS) $(LIB_KQED_GPU) $(LIB_KQED)
LIBS_CPU=$(LIBS) $(LIB_KQED)
DEVICE_LIBS=$(LIB_KQED_GPU)
DEVICE_LIB_DEPS=$(kqedcudadir)/lib/libKQED_cuda.a $(kqeddir)/lib/libKQED.a

LDFLAGS = $(LIBS_CPU) -fopenmp
LDFLAGS_GPU = $(LIBS_GPU) -fopenmp


LINK = $(CXX) -o $@
COMPILE = ${CXX} $(INCLUDE_CPU) -o $@ ${CXXFLAGS}
COMPILE_GPU = ${CXX} $(INCLUDE_GPU) -o $@ ${CXXFLAGS_GPU}
CUDA_COMPILE = ${NVCC} ${NVCCFLAGS} $(INCLUDE_GPU) -dc -o $@ --compiler-options="${CXXFLAGS_GPU}"
CUDA_LINK = ${NVCC} ${NVCCFLAGS} $(INCLUDE_GPU) -dlink -o $@ --compiler-options="${CXXFLAGS_GPU}" --ptxas-options="--allow-expensive-optimizations false"

MODULES = DML_crc32 dml getopt cvc_utils cvc_geometry mpi_init io io_utils propagator_io read_input_parser_cvc \
	  get_index gauge_io contractions_io ranlxd ranlxs Q_phi set_default \
	  Q_clover_phi gsp prepare_source matrix_init project make_x_orbits contract_cvc_tensor smearing_techniques \
	  prepare_propagator scalar_products clover rotations gitversion group_projection gamma dummy_solver contract_factorized \
	  contract_loop uwerr incomp_gamma derived_quantities
CUDA_MODULES = cuda_lattice

MODULES_OPT = kernels
CUDA_MODULES_OPT = kernels

HEADERS = getopt cvc_complex cvc_geometry cvc_linalg iblas icontract cvc_utils default_input_values dml global io io_utils \
	  mpi_init propagator_io read_input_parser get_index gauge_io contractions_io \
	  ranlxd ranlxs Q_phi set_default laplace_linalg Q_clover_phi \
	  gsp prepare_source matrix_init project make_x_orbits contract_cvc_tensor smearing_techniques \
	  prepare_propagator scalar_products clover rotations group_projection \
          set_cubic_group_double_cover_elements set_cubic_group_double_cover_identification_table set_cubic_group_elements \
          set_cubic_group_elements_v2 gamma gamma_mult_table clebsch_gordan \
	  table_init_z table_init_d table_init_i table_init_c table_init_asym_z group_projection_applicator dummy_solver contract_factorized \
	  contract_loop uwerr incomp_gamma derived_quantities cuda_lattice


# PROGRAM = hlbl_mII_contract_test
PROGRAM = hlbl_mII_invert_contract
# hlbl_2p2_invert_contract



all: dep $(PROGRAM) $(addsuffix .gpu,${PROGRAM})


# ##########

${srcdir}/read_input_parser_cvc.cpp: ${srcdir}/read_input_parser_cvc.l
	${LEX} -P cvc_ -i -t $< > ${srcdir}/read_input_parser_cvc.cpp

$(addsuffix .d,$(MODULES)): %.d: ${srcdir}/%.cpp
	 @ $(CCDEP) ${DEPFLAGS} ${INCLUDE} $< > $@
	
kernels_cpu.d: ${srcdir}/hlbl_kernels/kernels.cpp
	 @ $(CCDEP) ${DEPFLAGS} ${INCLUDE_CPU} $< > $@
 
$(addsuffix .d,$(CUDA_MODULES)): %.d: ${srcdir}/%.cu
	 @ $(CCDEP) ${DEPFLAGS} ${INCLUDE} $< > $@

kernels_gpu.d: ${srcdir}/hlbl_kernels/kernels.cu
	 @ $(CCDEP) ${DEPFLAGS} ${INCLUDE_GPU} $< > $@

$(addsuffix .d,$(PROGRAM)): %.d: ${srcdir}/%.cpp
	 @ $(CCDEP) ${DEPFLAGS} ${INCLUDE} $< > $@

dep: $(addsuffix .d,$(MODULES) $(CUDA_MODULES) ${PROGRAM}) kernels_cpu.d kernels_gpu.d

$(addsuffix .o,${MODULES}): %.o: ${srcdir}/%.cpp $(addprefix ${srcdir}/, $(addsuffix .h, ${HEADERS})) %.d
	${COMPILE} ${OPTARGS} -c $<

kernels_cpu.o: ${srcdir}/hlbl_kernels/kernels.cpp $(addprefix ${srcdir}/, $(addsuffix .h, ${HEADERS}))  $(srcdir)/hlbl_kernels/kernels.h kernels_cpu.d
	${COMPILE} ${OPTARGS} -c $<

$(addsuffix .o,${CUDA_MODULES}): %.o: ${srcdir}/%.cu $(addprefix ${srcdir}/, $(addsuffix .h, ${HEADERS})) %.d
	${CUDA_COMPILE} ${OPTARGS} -c $<

kernels_gpu.o: ${srcdir}/hlbl_kernels/kernels.cu $(addprefix ${srcdir}/, $(addsuffix .h, ${HEADERS})) kernels_gpu.d $(srcdir)/hlbl_kernels/kernels.cuh kernels_gpu.d
	${CUDA_COMPILE} ${OPTARGS} -c $<

$(addsuffix .o,${PROGRAM}): %.o: ${srcdir}/%.cpp %.d
	${COMPILE} ${OPTARGS} -c $< 

$(addsuffix .gpu.o,${PROGRAM}): %.gpu.o: ${srcdir}/%.cpp %.d
	${COMPILE_GPU} ${OPTARGS} -c $<

${PROGRAM}: %: %.o ${srcdir}/gitversion.cpp $(addsuffix .o,${MODULES}) kernels_cpu.o
	${LINK}  $(addsuffix .o,${MODULES}) kernels_cpu.o $@.o $(LDFLAGS)

$(addsuffix .gpu.dlink,${PROGRAM}): %.gpu.dlink: %.gpu.o ${srcdir}/gitversion.cpp $(addsuffix .o,${MODULES} ${CUDA_MODULES}) $(DEVICE_LIB_DEPS)
	${CUDA_LINK} kernels_gpu.o $(addsuffix .o,${MODULES} ${CUDA_MODULES}) $(DEVICE_LIBS) $<

$(addsuffix .gpu,${PROGRAM}): %.gpu: %.gpu.o %.gpu.dlink ${srcdir}/gitversion.cpp $(addsuffix .o,${MODULES} ${CUDA_MODULES}) kernels_gpu.o
	${LINK}  $(addsuffix .o,${MODULES} ${CUDA_MODULES}) kernels_gpu.o $@.o $@.dlink $(LDFLAGS_GPU)

${srcdir}/gitversion.cpp: ${srcdir}/.git/HEAD ${srcdir}/.git/index
	echo "namespace cvc { const char *g_gitversion = \"$(shell cd ${srcdir} && git rev-parse HEAD && cd -)\"; }" > ${srcdir}/gitversion.cpp

# ##########


clean:
	rm -f *~ *.o *.d *.dlink $(PROGRAM) $(addsuffix .gpu,${PROGRAM}) ${srcdir}/gitversion.cpp

.PHONY: clean

###########
