Imports System.Windows.Media.Media3D
Imports System.Threading.Tasks

Public Class TLMGrid

#Region "Initialize"

    ' Gives more weight to smoothing than to velocity
    Private _ptBuffer1 As Point3DCollection
    Private _ptBuffer2 As Point3DCollection
    Private _triangleIndices As Int32Collection

    ' These two pointers will swap, pointing to ptBuffer1/ptBuffer2 as we cycle the buffers
    Private _currBuffer As Point3DCollection
    Private _oldBuffer As Point3DCollection

    'Create scatter and incoming pressure matrix
    'The special connections combine speed of sound changes and attinuation of sound in the media
    Private _
    IncomingEast(Dimensions, Dimensions), _
    IncomingNorth(Dimensions, Dimensions), _
    IncomingWest(Dimensions, Dimensions), _
    IncomingSouth(Dimensions, Dimensions), _
 _
    ScatteredEast(Dimensions, Dimensions), _
    ScatteredNorth(Dimensions, Dimensions), _
    ScatteredWest(Dimensions, Dimensions), _
    ScatteredSouth(Dimensions, Dimensions), _
 _
    IncomingSpecial(Dimensions, Dimensions), _
    ScatteredSpecial(Dimensions, Dimensions), _
 _
    SpeedOfSoundAdjustment(Dimensions, Dimensions), _
    SoundAttinuation(Dimensions, Dimensions) _
    As Decimal

    'Create space for the differt tytpes of sources
    Private Sources(Dimensions, Dimensions) As Boolean
    Private DeltaSources(Dimensions, Dimensions) As Boolean
    Private GaussianSources(Dimensions, Dimensions) As Boolean

    'Number of time steps needed for the set cut off frequency
    Private GaussianTimeSteps(Dimensions, Dimensions) As Integer

    'Create space for wall elements
    Private Walls(Dimensions, Dimensions) As Decimal

    ' Bool values that is used to improve speed by pre setting of time consuming calcualtions
    Private WallNeigbor(Dimensions, Dimensions) As Boolean
    Private ChangeInWeightVertical(Dimensions, Dimensions) As Boolean
    Private ChangeInWeightHorizontal(Dimensions, Dimensions) As Boolean
    Private ChangeInWeight(Dimensions, Dimensions) As Boolean

    'Store the current pressure
    Private CurrentPressure(Dimensions, Dimensions) As Decimal

    'Make reflection and transmission based on weight of medium
    Private MassOFMedium(Dimensions, Dimensions) As Decimal

    Sub New(ByVal DimensionsOfRectangle As Integer)
        Dimensions = DimensionsOfRectangle

        'You have changed the dimensions so
        ' ...
        ' ...
        ' ...

        ReDim _
        IncomingEast(Dimensions, Dimensions), _
        IncomingNorth(Dimensions, Dimensions), _
        IncomingWest(Dimensions, Dimensions), _
        IncomingSouth(Dimensions, Dimensions), _
        ScatteredEast(Dimensions, Dimensions), _
        IncomingSpecial(Dimensions, Dimensions), _
        MassOFMedium(Dimensions, Dimensions), _
        ScatteredSpecial(Dimensions, Dimensions), _
        ScatteredNorth(Dimensions, Dimensions), _
        ScatteredWest(Dimensions, Dimensions), _
        ScatteredSouth(Dimensions, Dimensions), _
        SpeedOfSoundAdjustment(Dimensions, Dimensions), _
        SoundAttinuation(Dimensions, Dimensions)

        ReDim Sources(Dimensions, Dimensions)
        ReDim DeltaSources(Dimensions, Dimensions)
        ReDim GaussianSources(Dimensions, Dimensions)
        ReDim GaussianTimeSteps(Dimensions, Dimensions)

        ReDim Walls(Dimensions, Dimensions)
        ReDim WallNeigbor(Dimensions, Dimensions)
        ReDim CurrentPressure(Dimensions, Dimensions)
        ReDim ChangeInWeightVertical(Dimensions, Dimensions)
        ReDim ChangeInWeightHorizontal(Dimensions, Dimensions)
        ReDim ChangeInWeight(Dimensions, Dimensions)

        'Store buffer for the 3D viewer
        _ptBuffer1 = New Point3DCollection(Dimensions * Dimensions)
        _ptBuffer2 = New Point3DCollection(Dimensions * Dimensions)
        _triangleIndices = New Int32Collection((Dimensions - 1) * (Dimensions - 1) * 2)

        'Create a 3D mesh and connect it to collection of 3D points
        InitializePointsAndTriangles()

        _currBuffer = _ptBuffer2
        _oldBuffer = _ptBuffer1
    End Sub

#End Region

#Region "Properties"

    ''' <summary>
    ''' Number of cells in x and y direction
    ''' </summary>
    ''' <remarks></remarks>
    Private pDimensions As Integer = 100
    Public Property Dimensions() As Integer
        Get
            Return pDimensions
        End Get
        Set(ByVal value As Integer)
            pDimensions = value
        End Set
    End Property

    ''' <summary>
    ''' Currently calcualted delta steps 
    ''' </summary>
    ''' <remarks></remarks>
    Private pCurrentTimeStep As Integer = 0
    Public Property CurrentTimeStep() As Integer
        Get
            Return pCurrentTimeStep
        End Get
        Set(ByVal value As Integer)
            pCurrentTimeStep = value
        End Set
    End Property

    'Speed of sound in air
    Private pC0 As Decimal = 343
    Public Property C0() As Decimal
        Get
            Return pC0
        End Get
        Set(ByVal value As Decimal)
            pC0 = value
        End Set
    End Property

    'Speed of sound in TLM model
    Public ReadOnly Property CT As Decimal
        Get
            Return C0 / Math.Sqrt(2)
        End Get
    End Property

    ''' <summary>
    ''' Are there different weights within the model?
    ''' </summary>
    ''' <remarks></remarks>
    Private pWallsWithReflectionAndTransmission As Boolean = False
    Public Property WallsWithReflectionAndTransmission() As Boolean
        Get
            Return pWallsWithReflectionAndTransmission
        End Get
        Set(ByVal value As Boolean)
            pWallsWithReflectionAndTransmission = value
        End Set
    End Property

    ''' <summary>
    ''' Calcualte with non linear propagation (the NonLinearParameter must also be set)
    ''' </summary>
    ''' <remarks></remarks>
    Private pNonLinearPropagation As Boolean = False
    Public Property NonLinearPropagation() As Boolean
        Get
            Return pNonLinearPropagation
        End Get
        Set(ByVal value As Boolean)
            pNonLinearPropagation = value
        End Set
    End Property

    ''' <summary>
    ''' A non linear propagation value 
    ''' </summary>
    ''' <remarks></remarks>
    Private pNonLineraParameter As Decimal = 1.2
    Public Property NonLinearParamter() As Decimal
        Get
            Return pNonLineraParameter
        End Get
        Set(ByVal value As Decimal)
            pNonLineraParameter = value
        End Set
    End Property

    ''' <summary>
    ''' Normal weight of Air 20 degrees
    ''' </summary>
    ''' <remarks></remarks>
    Private PRho0 As Decimal = 1.21D
    Public Property Roh0() As Decimal
        Get
            Return PRho0
        End Get
        Set(ByVal value As Decimal)
            PRho0 = value
        End Set
    End Property

    ''' <summary>
    ''' Currently used frequency in the model
    ''' </summary>
    ''' <remarks></remarks>
    Private pSourceFrequency As Double = 8
    Public Property SourceFrequency() As Double
        Get
            Return pSourceFrequency
        End Get
        Set(ByVal value As Double)
            pSourceFrequency = value
        End Set
    End Property

    ''' <summary>
    ''' Spacing between each node
    ''' </summary>
    ''' <remarks></remarks>
    Private pDeltaLenght As Double = 1
    Public Property DeltaLength() As Double
        Get
            Return pDeltaLenght
        End Get
        Set(ByVal value As Double)
            pDeltaLenght = value
        End Set
    End Property

    ''' <summary>
    ''' Abovce this frequency the model is no longer correct
    ''' </summary>
    ''' <value></value>
    ''' <returns></returns>
    ''' <remarks></remarks>
    Public ReadOnly Property CutOffFrequency() As Double
        Get
            Return CT / DeltaLength / 4
        End Get
    End Property

    ''' <summary>
    ''' If you are above this frequency caution on calucalted results is adviced
    ''' </summary>
    ''' <value></value>
    ''' <returns></returns>
    ''' <remarks></remarks>
    Public ReadOnly Property RecomendedMAximumFrequency() As Double
        Get
            Return CT / DeltaLength / 10
        End Get
    End Property

    ''' <summary>
    ''' Used for the gaussian filter, recommended value range for approximatly 4 to 5
    ''' </summary>
    ''' <remarks></remarks>
    Private pMhy As Double = 5
    Public Property Mhy() As Double
        Get
            Return pMhy
        End Get
        Set(ByVal value As Double)
            pMhy = value
        End Set
    End Property

    ''' <summary>
    ''' Used to normalize the gaussian filter 
    ''' </summary>
    ''' <value></value>
    ''' <returns></returns>
    ''' <remarks></remarks>
    Public ReadOnly Property GaussianFilterSum() As Double
        Get
            Return 1 + 4 * Math.Exp(-Mhy) + 4 * Math.Exp(-2 * Mhy)
        End Get
    End Property

    ''' <summary>
    ''' Is the Gaussian filter turen on?
    ''' </summary>
    ''' <remarks></remarks>
    Private pGaussianFilter As Boolean = False
    Public Property GaussianFilter() As Boolean
        Get
            Return pGaussianFilter
        End Get
        Set(ByVal value As Boolean)
            pGaussianFilter = value
        End Set
    End Property

    ''' <summary>
    ''' Should model incorparate wave attenuation or change in speed?
    ''' </summary>
    ''' <remarks></remarks>
    Private NoSpecialScattering_value As Boolean = True
    Public Property NoSpecialScattering() As Boolean
        Get
            Return NoSpecialScattering_value
        End Get
        Set(ByVal value As Boolean)
            NoSpecialScattering_value = value
        End Set
    End Property

    ''' <summary>
    ''' Value below the fraction of the Gaussian source (internal and calcualted from the frequency)
    ''' </summary>
    ''' <value></value>
    ''' <returns></returns>
    ''' <remarks></remarks>
    Public ReadOnly Property TestA() As Double
        Get
            Return ((4 - Math.Log(Math.Sqrt(2 * Math.PI))) * 2 * CT / (Math.Log10(Math.E) * Math.PI ^ 2 * SourceFrequency ^ 2))
        End Get
    End Property

    ''' <summary>
    ''' Calcuatled from the source frequency and the maximum allowed step from 0 (in this case 10^-3 is its first value)
    ''' </summary>
    ''' <value></value>
    ''' <returns></returns>
    ''' <remarks></remarks>
    Public ReadOnly Property GaussianPulsLength() As Double
        Get
            Return Math.Sqrt(TestA ^ 2 * ((3 - Math.Log10(TestA * Math.Sqrt(Math.PI))) / Math.Log10(Math.E)))
        End Get
    End Property

    ''' <summary>
    ''' Get the matrix of the current pressure
    ''' </summary>
    ''' <value></value>
    ''' <returns></returns>
    ''' <remarks></remarks>
    Public ReadOnly Property GetCurrentPressure() As Decimal(,)
        Get
            Return CurrentPressure
        End Get
    End Property

    ''' <summary>
    ''' The actual length and width of the TLM model
    ''' </summary>
    ''' <value></value>
    ''' <returns></returns>
    ''' <remarks></remarks>
    Public ReadOnly Property RealLength() As Double
        Get
            Return DeltaLength * Dimensions
        End Get
    End Property

    ''' <summary>
    ''' Relating the time step in the model with the real time
    ''' </summary>
    ''' <value></value>
    ''' <returns></returns>
    ''' <remarks></remarks>
    Public ReadOnly Property DeltaTime() As Double
        Get
            Return DeltaLength / CT
        End Get
    End Property

    ''' <summary>
    ''' The amplitude of the sources
    ''' </summary>
    ''' <remarks></remarks>
    Private pAmplitude As Decimal = 5
    Public Property Amplitude() As Decimal
        Get
            Return pAmplitude
        End Get
        Set(ByVal value As Decimal)
            pAmplitude = value
        End Set
    End Property
#End Region

#Region "Set computing properties"
    Dim OpenEndedReflection As Decimal = ((1 - SquareRoot(2D)) / (1 + SquareRoot(2D)))

    ''' <summary>
    ''' Calcualte the square root of a decimal by the use of Newtons method
    ''' </summary>
    ''' <param name="number"></param>
    ''' <returns></returns>
    ''' <remarks></remarks>
    Public Function SquareRoot(ByVal number As Decimal) As Decimal

        ' if (x < 0) throw new OverflowException("Cannot calculate square root from a negative number");
        Dim epsilon As Decimal = 0.0000000000000000000000000002D
        Dim previous As Decimal
        Dim current As Decimal = CDec(Math.Sqrt(CDbl(number)))
        Do
            previous = current
            If previous = 0 Then Return 0
            current = (previous + number / previous) / 2D
        Loop While (Math.Abs(previous - current) > epsilon)

        Return current
    End Function

    Public Sub SetNonlinearPRopagation(ByVal IsOn As Boolean)
        For i As Integer = 0 To Dimensions - 1
            For j As Integer = 0 To Dimensions - 1
                If IsOn Then
                    'To have non linear propagation the Speed of sound would have to be adjusted to create a new normal speed of sound
                    SpeedOfSoundAdjustment(i, j) = 1
                Else
                    'Resetting the speed of sound in the model
                    SpeedOfSoundAdjustment(i, j) = 0
                End If
            Next
        Next

        NonLinearPropagation = IsOn
    End Sub


    Public Sub SetReflectionaAndTrasmissionCoefficients(ByVal ReflectionsOn As Boolean)
        For i As Integer = 0 To Dimensions - 1
            For j As Integer = 0 To Dimensions - 1
                If ReflectionsOn Then
                    'MassOFMedium(i, j) = j * 5 + 1
                    If j > Dimensions / 2 And j > Dimensions / 4 Then
                        ' That is depended on rho (Kg/m^3). If this is 1.21 nothing is changed in the model
                        MassOFMedium(i, j) = 10
                    ElseIf j > Dimensions / 4 Then
                        MassOFMedium(i, j) = 5
                    Else
                        'The mass of normal air at 20 degree celcius
                        MassOFMedium(i, j) = 1.21
                    End If
                Else
                    'Removes reflection and transmission, back to default
                    MassOFMedium(i, j) = 0
                End If
            Next
        Next

        ' Set boolean values in order to make quicker checks of normal propagation without any change
        For i As Integer = 0 To Dimensions - 1
            For j As Integer = 0 To Dimensions - 1
                If j = 0 OrElse i = 0 OrElse j = Dimensions - 1 OrElse i = Dimensions - 1 Then
                    'On the borders

                Else
                    'In free space
                    If MassOFMedium(i, j) <> MassOFMedium(i + 1, j) Then

                        ChangeInWeightHorizontal(i, j) = True
                        ChangeInWeightHorizontal(i + 1, j) = True
                        ChangeInWeight(i, j) = True
                        ChangeInWeight(i + 1, j) = True
                    End If

                    If MassOFMedium(i, j) <> MassOFMedium(i, j + 1) Then
                        ChangeInWeightVertical(i, j) = True
                        ChangeInWeightVertical(i, j + 1) = True
                        ChangeInWeight(i, j) = True
                        ChangeInWeight(i, j + 1) = True
                    End If
                End If
            Next
        Next

        WallsWithReflectionAndTransmission = ReflectionsOn
    End Sub

    Public Sub SetAttenuationCoefficient(ByVal AtteniationCoefficient As Decimal)
        For i As Integer = 0 To Dimensions - 1
            For j As Integer = 0 To Dimensions - 1
                If AtteniationCoefficient <> 0 Then
                    'The value should under normal circumstances correspond to 7.07 * 10 ^ -2  (neper/lambda) which is a standard value for Air at 20 degrees celcius
                    'It is recommended that this is used together with Gaussian filtering to reduce errors.
                    ' Attenuation = math.Sqrt(2)/DeltaL * Math.Log((n+4)/(n+e+4))
                    SoundAttinuation(i, j) = AtteniationCoefficient
                Else
                    SoundAttinuation(i, j) = 0
                End If
            Next
        Next
        If AtteniationCoefficient <> 0 Then
            NoSpecialScattering = False
        Else
            NoSpecialScattering = True
        End If
    End Sub

    Public Sub SetDifferentSpeed(ByVal DifferentSpeed As Boolean)
        For i As Integer = 0 To Dimensions - 1
            For j As Integer = 0 To Dimensions - 1
                If DifferentSpeed Then
                    'MassOFMedium(i, j) = j * 5 + 1
                    If j > Dimensions / 2 Then
                        ' That is depended on rho (Kg/m^3). If this is 1.21 nothing is changed in the model
                        SpeedOfSoundAdjustment(i, j) = 0.3
                    Else
                        SpeedOfSoundAdjustment(i, j) = 0
                    End If
                Else
                    'Removes reflection and transmission, back to default
                    SpeedOfSoundAdjustment(i, j) = 0
                End If
            Next
        Next

    End Sub

    Public Sub SetGaussianSources(ByVal SourcePoints As List(Of Point))
        For Each p As Point In SourcePoints
            GaussianSources(p.X, p.Y) = True
            GaussianTimeSteps(p.X, p.Y) = -GaussianPulsLength
        Next
    End Sub

    Public Sub SetGaussianSources(ByVal P As Point)
        GaussianSources(P.X, P.Y) = True
        GaussianTimeSteps(P.X, P.Y) = -GaussianPulsLength
    End Sub


    Public Sub SetDeltaSources(ByVal SourcePoints As List(Of Point))
        For Each p As Point In SourcePoints
            DeltaSources(p.X, p.Y) = True
        Next
    End Sub

    Public Sub SetDeltaSources(ByVal P As Point)
        DeltaSources(P.X, P.Y) = True
    End Sub

    Public Sub SetSources(ByVal SourcePoints As List(Of Point))
        For Each p As Point In SourcePoints
            Sources(p.X, p.Y) = True
        Next
    End Sub

    Public Sub SetSources(ByVal P As Point)
        Sources(P.X, P.Y) = True
    End Sub

    Dim SumSTep As Decimal = 0

    Public Sub SetWalls(ByVal WallPoints As List(Of Point), ByVal ReflectionFactor As Decimal)
        Dim R As Decimal = ((1 + ReflectionFactor) - SquareRoot(2) * (1 - ReflectionFactor)) / ((1 + ReflectionFactor) + SquareRoot(2) * (1 - ReflectionFactor))
        For Each p As Point In WallPoints
            Walls(p.X, p.Y) = R

            WallNeigbor(p.X, p.Y) = True

            If p.X <> 0 Then
                WallNeigbor(p.X - 1, p.Y) = True
            End If

            If p.X <> Dimensions - 1 Then
                WallNeigbor(p.X + 1, p.Y) = True
            End If

            If p.Y <> 0 Then
                WallNeigbor(p.X, p.Y - 1) = True
            End If

            If p.Y <> 0 Then
                WallNeigbor(p.X, p.Y + 1) = True
            End If
        Next
    End Sub
#End Region

#Region "Computations"

    Public Sub CalculateNextTimeStep()

        'Scatter wave
        Parallel.For(0, Dimensions, Sub(x)
                                        For y As Integer = 0 To Dimensions - 1
                                            SetSourcesInModel(x, y)

                                            If NoSpecialScattering Then
                                                ' The nodes have no sound attinuation or changes in speed
                                                NormalScatter(x, y)
                                            Else
                                                ' The nodes might have changes in speed, sound attinuation or Non linear propagation
                                                SpecialScatter(x, y)
                                            End If
                                        Next
                                    End Sub)

        'Propagate wave
        Parallel.For(0, Dimensions, Sub(x)
                                        For y As Integer = 0 To Dimensions - 1

                                            If Not (x = 0 Or y = 0 Or x = Dimensions - 1 Or y = Dimensions - 1) Then
                                                'Its not on the boandries 

                                                If Not (WallNeigbor(x, y)) Then
                                                    'its in the free field
                                                    If Not WallsWithReflectionAndTransmission Then
                                                        ' Has no other obsticles resulting from different masses
                                                        ' in neighboring cells 
                                                        FreeFieldPropagation(x, y)
                                                    Else
                                                        ' Calcualte reflections from different masses
                                                        ' in neighboring cells 
                                                        ReflectionAndTransmissiveWallPropagation(x, y)
                                                    End If
                                                Else
                                                    'Calculate backscatter with no transmission
                                                    ReflectionWallPropagation(x, y)
                                                End If
                                            Else
                                                ' We are on the boundaries
                                                If WallsWithReflectionAndTransmission Then
                                                    'Nodes at the end can have neighboring walls that can have both reflection and trasmission
                                                    FreeFieldBorderWithReflectionsAndTransmission(x, y)
                                                Else
                                                    'Nodes at the end can have neighboring walls that can have only reflection 
                                                    FreeFieldBorderWithReflections(x, y)
                                                End If
                                            End If

                                            ' Calcuatle the current pressure
                                            CurrentPressure(x, y) = GetCurrentPressureComputation(x, y)
                                        Next
                                    End Sub)

        'Apply the Gaussian filter on all the pressure points
        ' (if you have just one spot were you are interested in the sound pressure 
        ' it is enough to only filter that point)
        If GaussianFilter Then
            GaussianFilterCalcualtions()
        End If

        ' The delta time step in the simulation is increased
        CurrentTimeStep += 1

    End Sub

#Region "Sources"
    Private Sub SetSourcesInModel(ByVal x As Integer, ByVal y As Integer)
        If Sources(x, y) Then
            Dim Sine As Decimal = Amplitude * Math.Sin(Math.PI * CurrentTimeStep * DeltaLength * SourceFrequency / CT)
            IncomingEast(x, y) = Sine + IncomingEast(x, y)
            IncomingWest(x, y) = Sine + IncomingWest(x, y)
            IncomingSouth(x, y) = Sine + IncomingSouth(x, y)
            IncomingNorth(x, y) = Sine + IncomingNorth(x, y)
        End If

        If DeltaSources(x, y) Then
            IncomingEast(x, y) = Amplitude + IncomingEast(x, y)
            IncomingWest(x, y) = Amplitude + IncomingWest(x, y)
            IncomingSouth(x, y) = Amplitude + IncomingSouth(x, y)
            IncomingNorth(x, y) = Amplitude + IncomingNorth(x, y)
            DeltaSources(x, y) = False
        End If

        If GaussianSources(x, y) Then
            Dim Gauss As Double = TestA
            Gauss = Math.Exp(-(GaussianTimeSteps(x, y)) ^ 2 / TestA)

            IncomingEast(x, y) = Amplitude * Gauss + IncomingEast(x, y)
            IncomingWest(x, y) = Amplitude * Gauss + IncomingWest(x, y)
            IncomingSouth(x, y) = Amplitude * Gauss + IncomingSouth(x, y)
            IncomingNorth(x, y) = Amplitude * Gauss + IncomingNorth(x, y)
            GaussianTimeSteps(x, y) += 1
            If GaussianTimeSteps(x, y) = CInt(GaussianPulsLength) Then
                GaussianSources(x, y) = False
            End If
        End If
    End Sub
#End Region

#Region "Scatter"
    Private Sub NormalScatter(ByVal x As Integer, ByVal y As Integer)

        Dim IEast, INorth, ISouth, IWest As Decimal
        IEast = IncomingEast(x, y)
        INorth = IncomingNorth(x, y)
        ISouth = IncomingSouth(x, y)
        IWest = IncomingWest(x, y)


        ScatteredNorth(x, y) = 0.5 * (IEast - INorth + IWest + ISouth)
        ScatteredEast(x, y) = 0.5 * (-IEast + INorth + IWest + ISouth)
        ScatteredWest(x, y) = 0.5 * (IEast + INorth - IWest + ISouth)
        ScatteredSouth(x, y) = 0.5 * (IEast + INorth + IWest - ISouth)
    End Sub

    Private Sub SpecialScatter(ByVal x As Integer, ByVal y As Integer)

        Dim e, n As Decimal
        n = SpeedOfSoundAdjustment(x, y)

        If NonLinearPropagation Then
            'Propagation is dependent of current pressure
            Dim DeltaN As Decimal = -((2 * n + 8) * NonLinearParamter * CurrentPressure(x, y) / (Roh0 * C0 ^ 2))
            n += DeltaN
        End If

        e = SoundAttinuation(x, y)

        Dim AmplitudeFactor, BackScatterFactor As Decimal
        Dim IEast, INorth, ISouth, IWest, ISpecial As Decimal
        IEast = IncomingEast(x, y)
        INorth = IncomingNorth(x, y)
        ISouth = IncomingSouth(x, y)
        IWest = IncomingWest(x, y)
        ISpecial = IncomingSpecial(x, y)

        AmplitudeFactor = (2 / (n + e + 4))
        BackScatterFactor = (-1 - (n + e) / 2)

        ScatteredNorth(x, y) = AmplitudeFactor * (IEast + BackScatterFactor * INorth + IWest + ISouth + n * ISpecial)
        ScatteredEast(x, y) = AmplitudeFactor * (BackScatterFactor * IEast + INorth + IWest + ISouth + n * ISpecial)
        ScatteredWest(x, y) = AmplitudeFactor * (IEast + INorth + BackScatterFactor * IWest + ISouth + n * ISpecial)
        ScatteredSouth(x, y) = AmplitudeFactor * (IEast + INorth + IWest + BackScatterFactor * ISouth + n * ISpecial)
        ScatteredSpecial(x, y) = AmplitudeFactor * (IEast + INorth + IWest + ISouth + (n - e - 4) / 2 * ISpecial)
    End Sub
#End Region

#Region "Propagation of wave"
    Private Sub FreeFieldPropagation(ByVal x As Integer, ByVal y As Integer)
        IncomingEast(x, y) = ScatteredWest(x + 1, y)
        IncomingNorth(x, y) = ScatteredSouth(x, y + 1)
        IncomingWest(x, y) = ScatteredEast(x - 1, y)
        IncomingSouth(x, y) = ScatteredNorth(x, y - 1)
    End Sub

    Private Sub ReflectionAndTransmissiveWallPropagation(ByVal x As Integer, ByVal y As Integer)
        If Not ChangeInWeight(x, y) Then
            IncomingEast(x, y) = ScatteredWest(x + 1, y)
            IncomingNorth(x, y) = ScatteredSouth(x, y + 1)
            IncomingWest(x, y) = ScatteredEast(x - 1, y)
            IncomingSouth(x, y) = ScatteredNorth(x, y - 1)
        Else
            If ChangeInWeightHorizontal(x + 1, y) Then ' MassOFMedium(x, y) <> MassOFMedium(x + 1, y) Then
                Dim R, T As Decimal
                R = (MassOFMedium(x + 1, y) - MassOFMedium(x, y)) / (MassOFMedium(x + 1, y) + MassOFMedium(x, y))
                T = 1 + R
                IncomingEast(x, y) = R * ScatteredEast(x, y) + T * ScatteredWest(x + 1, y)
            Else
                IncomingEast(x, y) = ScatteredWest(x + 1, y)
            End If

            If ChangeInWeightVertical(x, y + 1) Then 'MassOFMedium(x, y) <> MassOFMedium(x, y + 1) Then
                Dim R, T As Decimal
                R = (MassOFMedium(x, y + 1) - MassOFMedium(x, y)) / (MassOFMedium(x, y + 1) + MassOFMedium(x, y))
                T = 1 + R
                IncomingNorth(x, y) = R * ScatteredNorth(x, y) + T * ScatteredSouth(x, y + 1)

            Else
                IncomingNorth(x, y) = ScatteredSouth(x, y + 1)
            End If


            If ChangeInWeightHorizontal(x - 1, y) Then 'MassOFMedium(x, y) <> MassOFMedium(x - 1, y) Then
                Dim R, T As Decimal
                R = (MassOFMedium(x - 1, y) - MassOFMedium(x, y)) / (MassOFMedium(x - 1, y) + MassOFMedium(x, y))
                T = 1 + R
                IncomingWest(x, y) = R * ScatteredWest(x, y) + T * ScatteredEast(x - 1, y)

            Else
                IncomingWest(x, y) = ScatteredEast(x - 1, y)
            End If


            If ChangeInWeightVertical(x, y - 1) Then 'MassOFMedium(x, y) <> MassOFMedium(x, y - 1) Then
                Dim R, T As Decimal
                R = (MassOFMedium(x, y - 1) - MassOFMedium(x, y)) / (MassOFMedium(x, y - 1) + MassOFMedium(x, y))
                T = 1 + R
                IncomingSouth(x, y) = R * ScatteredSouth(x, y) + T * ScatteredNorth(x, y - 1)

            Else
                IncomingSouth(x, y) = ScatteredNorth(x, y - 1)
            End If
        End If
    End Sub

    Private Sub ReflectionWallPropagation(ByVal x As Integer, ByVal y As Integer)
        If WallNeigbor(x + 1, y) Then
            IncomingEast(x, y) = Walls(x + 1, y) * ScatteredEast(x, y)
        Else
            IncomingEast(x, y) = ScatteredWest(x + 1, y)
        End If

        If WallNeigbor(x, y + 1) Then
            IncomingNorth(x, y) = Walls(x, y + 1) * ScatteredNorth(x, y)
        Else
            IncomingNorth(x, y) = ScatteredSouth(x, y + 1)
        End If

        If WallNeigbor(x - 1, y) Then
            IncomingWest(x, y) = Walls(x - 1, y) * ScatteredWest(x, y)
        Else
            IncomingWest(x, y) = ScatteredEast(x - 1, y)
        End If

        If WallNeigbor(x, y - 1) Then
            IncomingSouth(x, y) = Walls(x, y - 1) * ScatteredSouth(x, y)
        Else
            IncomingSouth(x, y) = ScatteredNorth(x, y - 1)
        End If
    End Sub

    Private Sub FreeFieldBorderWithReflections(ByVal x As Integer, ByVal y As Integer)
        'Its on the borders
        If x = 0 And y = 0 Then
            'Compared to the free field propagation x-1 and y-1 is illegal
            'This is the top left corner
            If Walls(x + 1, y) = 0 Then
                IncomingEast(x, y) = ScatteredWest(x + 1, y)
            Else
                IncomingEast(x, y) = Walls(x + 1, y) * ScatteredEast(x, y)
            End If


            If Walls(x, y + 1) = 0 Then
                IncomingNorth(x, y) = ScatteredSouth(x, y + 1)
            Else
                IncomingNorth(x, y) = Walls(x, y + 1) * ScatteredNorth(x, y)
            End If

            IncomingWest(x, y) = OpenEndedReflection * ScatteredWest(x, y)
            IncomingSouth(x, y) = OpenEndedReflection * ScatteredSouth(x, y)

        ElseIf x = 0 And y = Dimensions - 1 Then
            'Compared to the free field propagation x-1 and y+1 is illegal
            'This i sthe top right side
            If Walls(x + 1, y) = 0 Then
                IncomingEast(x, y) = ScatteredWest(x + 1, y)
            Else
                IncomingEast(x, y) = Walls(x + 1, y) * ScatteredEast(x, y)
            End If

            IncomingNorth(x, y) = OpenEndedReflection * ScatteredNorth(x, y)
            IncomingWest(x, y) = OpenEndedReflection * ScatteredWest(x, y)

            If Walls(x, y - 1) = 0 Then
                IncomingSouth(x, y) = ScatteredNorth(x, y - 1)
            Else
                IncomingSouth(x, y) = Walls(x, y - 1) * ScatteredSouth(x, y)
            End If


        ElseIf x = Dimensions - 1 And y = 0 Then
            'Compared to the free field propagation x+1 and y-1 is illegal
            'This is the bottom left corner
            IncomingEast(x, y) = OpenEndedReflection * ScatteredEast(x, y)
            If Walls(x, y + 1) = 0 Then
                IncomingNorth(x, y) = ScatteredSouth(x, y + 1)
            Else
                IncomingNorth(x, y) = Walls(x, y + 1) * ScatteredNorth(x, y)
            End If

            If Walls(x - 1, y) = 0 Then
                IncomingWest(x, y) = ScatteredEast(x - 1, y)
            Else
                IncomingWest(x, y) = Walls(x - 1, y) * ScatteredWest(x, y)
            End If

            IncomingSouth(x, y) = OpenEndedReflection * ScatteredSouth(x, y)

        ElseIf x = Dimensions - 1 And y = Dimensions - 1 Then
            'Compared to the free field propagation x+1 and y+1 is illegal
            'This is the Bottom right corner
            IncomingEast(x, y) = OpenEndedReflection * ScatteredEast(x, y)
            IncomingNorth(x, y) = OpenEndedReflection * ScatteredNorth(x, y)
            If Walls(x - 1, y) = 0 Then
                IncomingWest(x - 1, y) = ScatteredEast(x - 1, y)
            Else
                IncomingWest(x - 1, y) = Walls(x - 1, y) * ScatteredWest(x, y)
            End If

            If Walls(x, y - 1) = 0 Then
                IncomingSouth(x, y) = ScatteredNorth(x, y - 1)
            Else
                IncomingSouth(x, y) = Walls(x, y - 1) * ScatteredSouth(x, y)
            End If

        ElseIf x = 0 Then
            'Compared to the free field propagation x-1 is illegal
            'This is the top line
            If Walls(x + 1, y) = 0 Then
                IncomingEast(x, y) = ScatteredWest(x + 1, y)
            Else
                IncomingEast(x, y) = Walls(x + 1, y) * ScatteredEast(x, y)
            End If
            If Walls(x, y + 1) = 0 Then
                IncomingNorth(x, y) = ScatteredSouth(x, y + 1)
            Else
                IncomingNorth(x, y) = Walls(x, y + 1) * ScatteredNorth(x, y)
            End If
            If Walls(x, y - 1) = 0 Then
                IncomingSouth(x, y) = ScatteredNorth(x, y - 1)
            Else
                IncomingSouth(x, y) = Walls(x, y - 1) * ScatteredSouth(x, y)
            End If

            IncomingWest(x, y) = OpenEndedReflection * ScatteredWest(x, y)

        ElseIf y = 0 Then
            'Compared to the free field propagation y-1 is illegal
            IncomingSouth(x, y) = OpenEndedReflection * ScatteredSouth(x, y)

            If Walls(x + 1, y) = 0 Then
                IncomingEast(x, y) = ScatteredWest(x + 1, y)
            Else
                IncomingEast(x, y) = Walls(x + 1, y) * ScatteredEast(x, y)
            End If
            If Walls(x, y + 1) = 0 Then
                IncomingNorth(x, y) = ScatteredSouth(x, y + 1)
            Else
                IncomingNorth(x, y) = Walls(x, y + 1) * ScatteredNorth(x, y)
            End If
            If Walls(x - 1, y) = 0 Then
                IncomingWest(x, y) = ScatteredEast(x - 1, y)
            Else
                IncomingWest(x, y) = Walls(x - 1, y) * ScatteredWest(x, y)
            End If

        ElseIf y = Dimensions - 1 Then
            'Compared to the free field propagation y+1 is illegal

            IncomingNorth(x, y) = OpenEndedReflection * ScatteredNorth(x, y)

            If Walls(x + 1, y) = 0 Then
                IncomingEast(x, y) = ScatteredWest(x + 1, y)
            Else
                IncomingEast(x, y) = Walls(x + 1, y) * ScatteredEast(x, y)
            End If
            If Walls(x - 1, y) = 0 Then
                IncomingWest(x, y) = ScatteredEast(x - 1, y)
            Else
                IncomingWest(x, y) = Walls(x - 1, y) * ScatteredWest(x, y)
            End If
            If Walls(x, y - 1) = 0 Then
                IncomingSouth(x, y) = ScatteredNorth(x, y - 1)
            Else
                IncomingSouth(x, y) = Walls(x, y - 1) * ScatteredSouth(x, y)
            End If

        ElseIf x = Dimensions - 1 Then
            'Compared to the free field propagation x+1 is illegal
            IncomingEast(x, y) = OpenEndedReflection * ScatteredEast(x, y)

            If Walls(x, y + 1) = 0 Then
                IncomingNorth(x, y) = ScatteredSouth(x, y + 1)
            Else
                IncomingNorth(x, y) = Walls(x, y + 1) * ScatteredNorth(x, y)
            End If
            If Walls(x - 1, y) = 0 Then
                IncomingWest(x, y) = ScatteredEast(x - 1, y)
            Else
                IncomingWest(x, y) = Walls(x - 1, y) * ScatteredWest(x, y)
            End If
            If Walls(x, y - 1) = 0 Then
                IncomingSouth(x, y) = ScatteredNorth(x, y - 1)
            Else
                IncomingSouth(x, y) = Walls(x, y - 1) * ScatteredSouth(x, y)
            End If

        End If

    End Sub

    Private Sub FreeFieldBorderWithReflectionsAndTransmission(ByVal x As Integer, ByVal y As Integer)
        'Its on the borders
        If x = 0 And y = 0 Then
            'Compared to the free field propagation x-1 and y-1 is illegal
            'This is the top left corner
            If MassOFMedium(x, y) <> MassOFMedium(x + 1, y) Then
                Dim R, T As Decimal
                R = (MassOFMedium(x + 1, y) - MassOFMedium(x, y)) / (MassOFMedium(x + 1, y) + MassOFMedium(x, y))
                T = 1 + R
                IncomingEast(x, y) = R * ScatteredEast(x, y) + T * ScatteredWest(x + 1, y)
            Else
                IncomingEast(x, y) = ScatteredWest(x + 1, y)
            End If


            If MassOFMedium(x, y) <> MassOFMedium(x, y + 1) Then
                Dim R, T As Decimal
                R = (MassOFMedium(x, y + 1) - MassOFMedium(x, y)) / (MassOFMedium(x, y + 1) + MassOFMedium(x, y))
                T = 1 + R
                IncomingNorth(x, y) = R * ScatteredNorth(x, y) + T * ScatteredSouth(x, y + 1)

            Else
                IncomingNorth(x, y) = ScatteredSouth(x, y + 1)
            End If

            IncomingWest(x, y) = OpenEndedReflection * ScatteredWest(x, y)
            IncomingSouth(x, y) = OpenEndedReflection * ScatteredSouth(x, y)

        ElseIf x = 0 And y = Dimensions - 1 Then
            'Compared to the free field propagation x-1 and y+1 is illegal
            'This i sthe top right side
            If MassOFMedium(x, y) <> MassOFMedium(x + 1, y) Then
                Dim R, T As Decimal
                R = (MassOFMedium(x + 1, y) - MassOFMedium(x, y)) / (MassOFMedium(x + 1, y) + MassOFMedium(x, y))
                T = 1 + R
                IncomingEast(x, y) = R * ScatteredEast(x, y) + T * ScatteredWest(x + 1, y)
            Else
                IncomingEast(x, y) = ScatteredWest(x + 1, y)
            End If

            IncomingNorth(x, y) = OpenEndedReflection * ScatteredNorth(x, y)
            IncomingWest(x, y) = OpenEndedReflection * ScatteredWest(x, y)


            If MassOFMedium(x, y) <> MassOFMedium(x, y - 1) Then
                Dim R, T As Decimal
                R = (MassOFMedium(x, y - 1) - MassOFMedium(x, y)) / (MassOFMedium(x, y - 1) + MassOFMedium(x, y))
                T = 1 + R
                IncomingSouth(x, y) = R * ScatteredSouth(x, y) + T * ScatteredNorth(x, y - 1)

            Else
                IncomingSouth(x, y) = ScatteredNorth(x, y - 1)
            End If


        ElseIf x = Dimensions - 1 And y = 0 Then
            'Compared to the free field propagation x+1 and y-1 is illegal
            'This is the bottom left corner
            IncomingEast(x, y) = OpenEndedReflection * ScatteredEast(x, y)
            If MassOFMedium(x, y) <> MassOFMedium(x, y + 1) Then
                Dim R, T As Decimal
                R = (MassOFMedium(x, y + 1) - MassOFMedium(x, y)) / (MassOFMedium(x, y + 1) + MassOFMedium(x, y))
                T = 1 + R
                IncomingNorth(x, y) = R * ScatteredNorth(x, y) + T * ScatteredSouth(x, y + 1)

            Else
                IncomingNorth(x, y) = ScatteredSouth(x, y + 1)
            End If

            If MassOFMedium(x, y) <> MassOFMedium(x - 1, y) Then
                Dim R, T As Decimal
                R = (MassOFMedium(x - 1, y) - MassOFMedium(x, y)) / (MassOFMedium(x - 1, y) + MassOFMedium(x, y))
                T = 1 + R
                IncomingWest(x, y) = R * ScatteredWest(x, y) + T * ScatteredEast(x - 1, y)

            Else
                IncomingWest(x, y) = ScatteredEast(x - 1, y)
            End If

            IncomingSouth(x, y) = OpenEndedReflection * ScatteredSouth(x, y)

        ElseIf x = Dimensions - 1 And y = Dimensions - 1 Then
            'Compared to the free field propagation x+1 and y+1 is illegal
            'This is the Bottom right corner
            IncomingEast(x, y) = OpenEndedReflection * ScatteredEast(x, y)
            IncomingNorth(x, y) = OpenEndedReflection * ScatteredNorth(x, y)
            If MassOFMedium(x, y) <> MassOFMedium(x - 1, y) Then
                Dim R, T As Decimal
                R = (MassOFMedium(x - 1, y) - MassOFMedium(x, y)) / (MassOFMedium(x - 1, y) + MassOFMedium(x, y))
                T = 1 + R
                IncomingWest(x, y) = R * ScatteredWest(x, y) + T * ScatteredEast(x - 1, y)

            Else
                IncomingWest(x, y) = ScatteredEast(x - 1, y)
            End If

            If MassOFMedium(x, y) <> MassOFMedium(x, y - 1) Then
                Dim R, T As Decimal
                R = (MassOFMedium(x, y - 1) - MassOFMedium(x, y)) / (MassOFMedium(x, y - 1) + MassOFMedium(x, y))
                T = 1 + R
                IncomingSouth(x, y) = R * ScatteredSouth(x, y) + T * ScatteredNorth(x, y - 1)

            Else
                IncomingSouth(x, y) = ScatteredNorth(x, y - 1)
            End If

        ElseIf x = 0 Then
            'Compared to the free field propagation x-1 is illegal
            'This is the top line
            If MassOFMedium(x, y) <> MassOFMedium(x + 1, y) Then
                Dim R, T As Decimal
                R = (MassOFMedium(x + 1, y) - MassOFMedium(x, y)) / (MassOFMedium(x + 1, y) + MassOFMedium(x, y))
                T = 1 + R
                IncomingEast(x, y) = R * ScatteredEast(x, y) + T * ScatteredWest(x + 1, y)
            Else
                IncomingEast(x, y) = ScatteredWest(x + 1, y)
            End If

            If MassOFMedium(x, y) <> MassOFMedium(x, y + 1) Then
                Dim R, T As Decimal
                R = (MassOFMedium(x, y + 1) - MassOFMedium(x, y)) / (MassOFMedium(x, y + 1) + MassOFMedium(x, y))
                T = 1 + R
                IncomingNorth(x, y) = R * ScatteredNorth(x, y) + T * ScatteredSouth(x, y + 1)

            Else
                IncomingNorth(x, y) = ScatteredSouth(x, y + 1)
            End If


            If MassOFMedium(x, y) <> MassOFMedium(x, y - 1) Then
                Dim R, T As Decimal
                R = (MassOFMedium(x, y - 1) - MassOFMedium(x, y)) / (MassOFMedium(x, y - 1) + MassOFMedium(x, y))
                T = 1 + R
                IncomingSouth(x, y) = R * ScatteredSouth(x, y) + T * ScatteredNorth(x, y - 1)

            Else
                IncomingSouth(x, y) = ScatteredNorth(x, y - 1)
            End If

            IncomingWest(x, y) = OpenEndedReflection * ScatteredWest(x, y)

        ElseIf y = 0 Then
            'Compared to the free field propagation y-1 is illegal
            IncomingSouth(x, y) = OpenEndedReflection * ScatteredSouth(x, y)

            If MassOFMedium(x, y) <> MassOFMedium(x + 1, y) Then
                Dim R, T As Decimal
                R = (MassOFMedium(x + 1, y) - MassOFMedium(x, y)) / (MassOFMedium(x + 1, y) + MassOFMedium(x, y))
                T = 1 + R
                IncomingEast(x, y) = R * ScatteredEast(x, y) + T * ScatteredWest(x + 1, y)
            Else
                IncomingEast(x, y) = ScatteredWest(x + 1, y)
            End If

            If MassOFMedium(x, y) <> MassOFMedium(x, y + 1) Then
                Dim R, T As Decimal
                R = (MassOFMedium(x, y + 1) - MassOFMedium(x, y)) / (MassOFMedium(x, y + 1) + MassOFMedium(x, y))
                T = 1 + R
                IncomingNorth(x, y) = R * ScatteredNorth(x, y) + T * ScatteredSouth(x, y + 1)

            Else
                IncomingNorth(x, y) = ScatteredSouth(x, y + 1)
            End If

            If MassOFMedium(x, y) <> MassOFMedium(x - 1, y) Then
                Dim R, T As Decimal
                R = (MassOFMedium(x - 1, y) - MassOFMedium(x, y)) / (MassOFMedium(x - 1, y) + MassOFMedium(x, y))
                T = 1 + R
                IncomingWest(x, y) = R * ScatteredWest(x, y) + T * ScatteredEast(x - 1, y)

            Else
                IncomingWest(x, y) = ScatteredEast(x - 1, y)
            End If

        ElseIf y = Dimensions - 1 Then
            'Compared to the free field propagation y+1 is illegal

            IncomingNorth(x, y) = OpenEndedReflection * ScatteredNorth(x, y)

            If MassOFMedium(x, y) <> MassOFMedium(x + 1, y) Then
                Dim R, T As Decimal
                R = (MassOFMedium(x + 1, y) - MassOFMedium(x, y)) / (MassOFMedium(x + 1, y) + MassOFMedium(x, y))
                T = 1 + R
                IncomingEast(x, y) = R * ScatteredEast(x, y) + T * ScatteredWest(x + 1, y)
            Else
                IncomingEast(x, y) = ScatteredWest(x + 1, y)
            End If

            If MassOFMedium(x, y) <> MassOFMedium(x - 1, y) Then
                Dim R, T As Decimal
                R = (MassOFMedium(x - 1, y) - MassOFMedium(x, y)) / (MassOFMedium(x - 1, y) + MassOFMedium(x, y))
                T = 1 + R
                IncomingWest(x, y) = R * ScatteredWest(x, y) + T * ScatteredEast(x - 1, y)

            Else
                IncomingWest(x, y) = ScatteredEast(x - 1, y)
            End If

            If MassOFMedium(x, y) <> MassOFMedium(x, y - 1) Then
                Dim R, T As Decimal
                R = (MassOFMedium(x, y - 1) - MassOFMedium(x, y)) / (MassOFMedium(x, y - 1) + MassOFMedium(x, y))
                T = 1 + R
                IncomingSouth(x, y) = R * ScatteredSouth(x, y) + T * ScatteredNorth(x, y - 1)

            Else
                IncomingSouth(x, y) = ScatteredNorth(x, y - 1)
            End If

        ElseIf x = Dimensions - 1 Then
            'Compared to the free field propagation x+1 is illegal
            IncomingEast(x, y) = OpenEndedReflection * ScatteredEast(x, y)

            If MassOFMedium(x, y) <> MassOFMedium(x, y + 1) Then
                Dim R, T As Decimal
                R = (MassOFMedium(x, y + 1) - MassOFMedium(x, y)) / (MassOFMedium(x, y + 1) + MassOFMedium(x, y))
                T = 1 + R
                IncomingNorth(x, y) = R * ScatteredNorth(x, y) + T * ScatteredSouth(x, y + 1)

            Else
                IncomingNorth(x, y) = ScatteredSouth(x, y + 1)
            End If

            If MassOFMedium(x, y) <> MassOFMedium(x - 1, y) Then
                Dim R, T As Decimal
                R = (MassOFMedium(x - 1, y) - MassOFMedium(x, y)) / (MassOFMedium(x - 1, y) + MassOFMedium(x, y))
                T = 1 + R
                IncomingWest(x, y) = R * ScatteredWest(x, y) + T * ScatteredEast(x - 1, y)

            Else
                IncomingWest(x, y) = ScatteredEast(x - 1, y)
            End If

            If MassOFMedium(x, y) <> MassOFMedium(x, y - 1) Then
                Dim R, T As Decimal
                R = (MassOFMedium(x, y - 1) - MassOFMedium(x, y)) / (MassOFMedium(x, y - 1) + MassOFMedium(x, y))
                T = 1 + R
                IncomingSouth(x, y) = R * ScatteredSouth(x, y) + T * ScatteredNorth(x, y - 1)

            Else
                IncomingSouth(x, y) = ScatteredNorth(x, y - 1)
            End If

        End If
    End Sub
#End Region

#Region "Pressure calcualtions"
    Private Function GetCurrentPressureComputation(ByVal x As Integer, ByVal y As Integer) As Decimal
        Dim TempSumPressure As Decimal = ScatteredEast(x, y) + ScatteredNorth(x, y) + ScatteredWest(x, y) + ScatteredSouth(x, y)
        Dim P As Decimal
        If NoSpecialScattering Then
            P = (0.5 * (TempSumPressure))
        Else
            Dim n, e, T1, T2, T3 As Decimal
            n = SpeedOfSoundAdjustment(x, y)
            e = SoundAttinuation(x, y)
            T1 = ((n + 4) / (n + e + 4))
            T2 = (2 / (n + 4))
            T3 = (2 * n / (n + 4))
            P = T1 * (T2 * (TempSumPressure) + (T3 * ScatteredSpecial(x, y)))
        End If
        Return P
    End Function


#End Region

#Region "Pressure filters"
    Private Sub GaussianFilterCalcualtions()
        Dim N0, N1, N2 As Double
        N1 = Math.Exp(-Mhy)
        N2 = Math.Exp(-2 * Mhy)
        N0 = (1 / GaussianFilterSum)
        Parallel.For(1, Dimensions - 1, Sub(x)
                                            For y As Integer = 1 To Dimensions - 2
                                                CurrentPressure(x, y) = N0 * (CurrentPressure(x, y) + _
                                                                                                   N1 * (CurrentPressure(x, y - 1) + CurrentPressure(x, y + 1) + CurrentPressure(x - 1, y) + CurrentPressure(x + 1, y)) _
                                                                                                   + N2 * (CurrentPressure(x - 1, y - 1) + CurrentPressure(x + 1, y - 1) + CurrentPressure(x - 1, y + 1) + CurrentPressure(x + 1, y + 1)))
                                            Next


                                        End Sub)
    End Sub
#End Region

#End Region

#Region "Reset and clear functions"
    Public Sub ClearAll()
        CurrentTimeStep = 0

        For i As Integer = 0 To Dimensions - 1
            For j As Integer = 0 To Dimensions - 1
                IncomingEast(i, j) = 0
                IncomingNorth(i, j) = 0
                IncomingWest(i, j) = 0
                IncomingSouth(i, j) = 0
                ScatteredEast(i, j) = 0
                ScatteredNorth(i, j) = 0
                ScatteredWest(i, j) = 0
                ScatteredSouth(i, j) = 0
                Sources(i, j) = False
                DeltaSources(i, j) = False
                GaussianSources(i, j) = False
                GaussianTimeSteps(i, j) = 0
                WallNeigbor(i, j) = False
                Walls(i, j) = 0
                CurrentPressure(i, j) = 0
            Next
        Next
    End Sub

    Public Sub Restart()
        CurrentTimeStep = 0

        For i As Integer = 0 To Dimensions - 1
            For j As Integer = 0 To Dimensions - 1
                IncomingEast(i, j) = 0
                IncomingNorth(i, j) = 0
                IncomingWest(i, j) = 0
                IncomingSouth(i, j) = 0
                ScatteredEast(i, j) = 0
                ScatteredNorth(i, j) = 0
                ScatteredWest(i, j) = 0
                ScatteredSouth(i, j) = 0

                GaussianTimeSteps(i, j) = 0

                CurrentPressure(i, j) = 0
            Next
        Next
    End Sub

#End Region

#Region "3D functions"

    ' ''' <summary>
    ' ''' Clear out points/triangles and regenerates
    ' ''' </summary>
    ' ''' <param name="grid"></param>
    Private Sub InitializePointsAndTriangles()
        _ptBuffer1.Clear()
        _ptBuffer2.Clear()
        _triangleIndices.Clear()

        Dim nCurrIndex As Integer = 0
        ' March through 1-D arrays

        For row As Integer = 0 To Dimensions - 1
            For col As Integer = 0 To Dimensions - 1
                ' In grid, X/Y values are just row/col numbers
                _ptBuffer1.Add(New Point3D(col, 0.0, row))

                ' Completing new square, add 2 triangles
                If (row > 0) AndAlso (col > 0) Then
                    ' Triangle 1
                    _triangleIndices.Add(nCurrIndex - Dimensions - 1)
                    _triangleIndices.Add(nCurrIndex)
                    _triangleIndices.Add(nCurrIndex - Dimensions)

                    ' Triangle 2
                    _triangleIndices.Add(nCurrIndex - Dimensions - 1)
                    _triangleIndices.Add(nCurrIndex - 1)
                    _triangleIndices.Add(nCurrIndex)
                End If

                nCurrIndex += 1
            Next
        Next

        ' 2nd buffer exists only to have 2nd set of Z values
        _ptBuffer2 = _ptBuffer1.Clone()
    End Sub

    Public Sub UpdateValues()
        Dim row As Integer = (((Dimensions) - 1.0))
        Dim col As Integer = ((Dimensions - 1.0))

        ' Change data 
        For ir As Integer = 0 To row
            For ic As Integer = 0 To (col)
                Dim pt As Point3D = _oldBuffer((ir * Dimensions) + ic)
                pt.Y = CurrentPressure(ir, ic)
                _oldBuffer((ir * Dimensions) + ic) = pt
            Next
        Next
        SwapBuffers()
    End Sub

    Public ReadOnly Property Points() As Point3DCollection
        Get
            Return _currBuffer
        End Get
    End Property

    Public ReadOnly Property TriangleIndices() As Int32Collection
        Get
            Return _triangleIndices
        End Get
    End Property

    Private Sub SwapBuffers()
        Dim temp As Point3DCollection = _currBuffer
        _currBuffer = _oldBuffer
        _oldBuffer = temp
    End Sub

#End Region

End Class
